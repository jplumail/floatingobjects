import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import rasterio
from tqdm.auto import tqdm as tq
from torch.utils.data import DataLoader, random_split

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from floatingobjects.visualization import calculate_fdi, ndvi_transform
from floatingobjects.data import FloatingSeaObjectDataset
from floatingobjects.train import predict_images
from floatingobjects.model import get_model


N_PIXELS_FOR_EACH_CLASS_FROM_IMAGE = 5
LABELS = ["no floating", "floating"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def sample_N_random(data, N):
    idxs = np.random.choice(np.arange(len(data)), min(len(data), N), replace=False)
    return data[idxs]


def aggregate_images(x, y):
    """
    aggregates images to pixel datasets
    x (image): (N_im x) D x H x W -> N x D where N randomly sampled pixels within image
    y (label): (N_im x) H x W -> N
    """
    N = N_PIXELS_FOR_EACH_CLASS_FROM_IMAGE

    N_images = x.shape[0] if len(x.shape) == 4 else 1
    floating_objects = x.swapaxes(0, 1)[:, y.astype(bool)].T  # 1: floating
    not_floating_objects = x.swapaxes(0, 1)[:, ~y.astype(bool)].T  # 0: no floating

    # use less if len(floating_objects) < N
    N_p = min(N*N_images, len(floating_objects))

    x_floating_objects = sample_N_random(floating_objects, N_p)
    y_floating_objects = np.ones(x_floating_objects.shape[0], dtype=int)

    x_not_floating_objects = sample_N_random(not_floating_objects, N_p)
    y_not_floating_objects = np.zeros(x_not_floating_objects.shape[0], dtype=int)

    x = np.concatenate([x_floating_objects, x_not_floating_objects], axis=0)
    y = np.concatenate([y_floating_objects, y_not_floating_objects], axis=0)
    return x, y


def draw_N_datapoints(dataset, N):
    idxs = np.random.randint(len(dataset), size=N)

    x = []
    y = []
    for idx in idxs:
        x_,y_,fid_ = dataset[idx]
        x.append(x_)
        y.append(y_)

    return np.stack(x, axis=0), np.stack(y, axis=0)


def feature_extraction_transform(x, y):
    x, y = aggregate_images(x, y)
    x = s2_to_ndvifdi(x)

    return x, y


def s2_to_ndvifdi(x):
    """
    x : array, shape ... x 12
    return : array, shape ... x 2
    """
    ndvi = ndvi_transform(x.T).T
    fdi = calculate_fdi(x.T).T
    return np.stack([fdi, ndvi], axis=-1)


bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
def s2_to_6bands(x):
    B02 = x[..., bands.index("B2")]
    B04 = x[..., bands.index("B4")]
    B06 = x[..., bands.index("B6")]
    B03 = x[..., bands.index("B3")]
    B08 = x[..., bands.index("B8")]
    B11 = x[..., bands.index("B11")]

    return np.stack([B02,B03,B04,B06,B08,B11], axis=-1)

def feature_extraction_transform_6(x, y):
    x, y = aggregate_images(x, y)
    x = s2_to_6bands(x)

    return x, y

def s2_to_ndvifdi_test(x, y):
    """
    x : array, shape B x 12 x H x W
    y : array, shape B x H x W
    return : array, shape B*H*W x 2
    """
    x_ = s2_to_ndvifdi(x.swapaxes(1,3))
    x_ = x_.reshape(-1, 2)
    return x_, y.reshape(-1)

def s2_to_6bands_test(x, y):
    """
    x : array, shape B x 12 x H x W
    y : array, shape B x H x W
    return : array, shape B*H*W x 2
    """
    x_ = s2_to_6bands(x.swapaxes(1,3))
    x_ = x_.reshape(-1, 6)
    return x_, y.reshape(-1)

def s2_to_12bands_test(x, y):
    """
    x : array, shape B x 12 x H x W
    y : array, shape B x H x W
    return : array, shape B*H*W x 12
    """
    x_ = x.swapaxes(0,1).reshape(12, -1)
    return x_.T, y.reshape(-1)

#############################################################################################################
data_path = "floatingobjects/data" # in our case, data is here
image_size = 128

threshold = 0.03
N_pixels = 20000
#seed = 1
train, test, metrics = True, True, True

transform_methods = {
    "train": {
        "indices": feature_extraction_transform,
        "6-bands": feature_extraction_transform_6,
        "12-bands": aggregate_images
    },
    "test": {
        "indices": s2_to_ndvifdi_test,
        "6-bands": s2_to_6bands_test,
        "12-bands": s2_to_12bands_test
    }
}

#folds = [1, 2]
folds = [1]
methods = ["indices", "12-bands", "6-bands"]
#methods = ["indices"]


# Create classical methods
clf_classes = {
    "SVM": make_pipeline(svm.SVC(C=30, gamma=0.001, cache_size=1000)),
    "NB": make_pipeline(GaussianNB()),
    "RF": make_pipeline(RandomForestClassifier(n_estimators=1000, max_depth=2, n_jobs=-1)),
    "HGB": make_pipeline(HistGradientBoostingClassifier())
}
clf_classes = {"NB": make_pipeline(GaussianNB())}

# Load deep learning models
model_names = ["unet", "manet"]
model_names = [] # no dl
models_dl = {
    f: {
        model_name: get_model(model_name, inchannels=12, pretrained=False).to(device).eval() for model_name in model_names
    } for f in folds
}
for f in folds:
    for model_name in models_dl[f]:
        path = f"models\{model_name}-posweight1-lr001-bs160-ep50-aug1-seed{f-1}.pth.tar"
        snapshot_file = torch.load(path, map_location=device)
        models_dl[f][model_name].load_state_dict(snapshot_file["model_state_dict"])

# Load ships classifiers
classifier_ships = {
    f: get_model("classifier", inchannels=12).to(device).eval() for f in folds
}
for f in folds:
    path = f"models\checkpoint-fold{f}.pt"
    snapshot_file = torch.load(path, map_location=device)
    classifier_ships[f].load_state_dict(snapshot_file["state_dict"])


def main():

    # Dictionnary containing all of the classifiers for each fold, method (12/6 bands...)
    clfs = {
        f: {
             m: {
                 clf_name : clone(clf_classes[clf_name]) for clf_name in clf_classes
             } for m in methods
        } for f in folds
    }

    #### Train
    if train:
        print("Training\n")
        for fold in tq(folds, desc="Folds"): # 1, 2
            # dataset for train with no feature extraction, features aggregation
            trainimagedataset = FloatingSeaObjectDataset(data_path, fold="train", foldn=fold, transform=None, output_size=image_size)
            validmagedataset = FloatingSeaObjectDataset(data_path, fold="val", foldn=fold, transform=None, output_size=image_size)

            x_train, y_train = draw_N_datapoints(trainimagedataset, N=100000)
            print(x_train.shape)
            x_test, y_test = draw_N_datapoints(validmagedataset, N=10000)

            for method in tq(methods, desc="Methods"): # "indices", "6-bands", "12-bands"

                transform_train = transform_methods["train"][method]
                x_train_m, y_train_m = transform_train(x_train, y_train)
                x_test_m, y_test_m = transform_train(x_test, y_test)

                for clf_name in tq(clf_classes, desc="Algorithms"):
                    clf = clfs[fold][method][clf_name]
                    clf.fit(x_train_m, y_train_m)
                    y_pred = clf.predict(x_test_m)
                    print(method + ", fold " + str(fold) + ", " + clf_name)
                    print(classification_report(y_test_m, y_pred, target_names=["water","floating objects"]))

    metrics_dir = "metrics/"
    threshold = 0.03
    if test:
        #### Test
        print("\nTest\n")

        #confusion matrix
        conf_mat = np.zeros((len(LABELS), len(LABELS)))

        from time import time

        # SVM, RF, NB and HGB
        for fold in tq(folds, desc="Folds"):
            dir_fold = os.path.join(metrics_dir, str(fold))
            os.makedirs(dir_fold, exist_ok=True)

            # dataset for validation with no feature extraction
            testimagedataset = FloatingSeaObjectDataset(data_path, fold="test", foldn=fold, transform=None, output_size=image_size)
            test_loader = DataLoader(testimagedataset, batch_size=1, shuffle=False, num_workers=4)

            # Iterate through "indices", "6-bands"...
            for d, method in tq(zip([12,6,2], methods), desc="Methods"):
                dir_method = os.path.join(dir_fold, method)
                os.makedirs(dir_method, exist_ok=True)

                transform_test = transform_methods["test"][method]

                y_trues = []
                # Iterate through dataset
                for idx, (image, y_true, _) in tq(enumerate(test_loader), total=len(test_loader)):
                    features, y_true = transform_test(image, y_true)
                    y_trues.append(y_true)

                    # Predict for each classifier
                    for k in clfs[fold][method]:
                        dir_clf = os.path.join(dir_method, k)
                        os.makedirs(dir_clf, exist_ok=True)
                        clf = clfs[fold][method][k]
                        y_pred = clf.predict(features)
                        torch.save(y_pred, os.path.join(dir_clf, f'y_pred_{idx}.pt'))
                    
                    if method == "12-bands":
                        image = image.float().to(device)
                        for k in models_dl[fold]:
                            dir_dl = os.path.join(dir_method, k, "no_classifier")
                            dir_dl_classifier = os.path.join(dir_method, k, "with_classifier")
                            os.makedirs(dir_dl, exist_ok=True)
                            os.makedirs(dir_dl_classifier, exist_ok=True)

                            model = models_dl[fold][k]
                            classifier = classifier_ships[fold]
                            with torch.no_grad():
                                logits = model(image)
                                output = torch.sigmoid(logits)[:,0]
                                mask = output > threshold
                                new_mask = classifier.sliding_windows(image, mask, threshold=0.01)
                            new_output = torch.clone(output)
                            new_output[~new_mask] = 0.
                            y_pred = output.view(-1).to("cpu").numpy()
                            new_y_pred = new_output.view(-1).to("cpu").numpy()
                            torch.save(y_pred, os.path.join(dir_dl, f'y_pred_{idx}.pt'))
                            torch.save(new_y_pred, os.path.join(dir_dl_classifier, f'y_pred_{idx}.pt'))
                y_trues = torch.stack(y_trues, dim=0)
                torch.save(y_trues, os.path.join(dir_fold, f'y_true.pt'))


        
        
    
    ######################################################################################
    # Check metrics

    threshold = 0.03
    N_pixels = 40000

    def load_y(path):
        return np.stack([torch.load(os.path.join(path,f)) for f in os.listdir(path)], axis=0)
    
    if metrics:
        for fold in folds:
            dir_fold = os.path.join(metrics_dir, str(fold))
            report = f"Fold {fold} report\n"
            fold_report_path = os.path.join(dir_fold, "report.txt")

            print("Fold :", fold)
            y_true = np.stack(torch.load(os.path.join(dir_fold, f'y_true.pt')), axis=0)
            y_true_flat = np.vstack(y_true).reshape(-1)
            idx_floating, = np.where(y_true_flat.astype(bool))
            idx_water, = np.where(~y_true_flat.astype(bool))
            idx_floating_choice = np.random.choice(idx_floating, N_pixels)
            idx_water_choice = np.random.choice(idx_water, N_pixels)
            idx = np.hstack([idx_floating_choice, idx_water_choice])
            for method in methods:
                print("Method :", method)
                report += f"Method : {method}\n"
                for k in clfs[fold][method]:
                    dir_clf = os.path.join(metrics_dir, str(fold), method, k)
                    y_pred = load_y(dir_clf)
                    
                    print("Model:", k)
                    y_pred_flat = np.vstack(y_pred).reshape(-1)
                    conf_mat = confusion_matrix(y_true_flat[idx], y_pred_flat[idx] > threshold)
                    print(conf_mat)
                    
                    report += f"Model : {k}\n"
                    report += str(conf_mat) + "\n"
                    report += classification_report(y_true_flat[idx], y_pred_flat[idx], target_names=["water","floating objects"]) + "\n"
                
                if method == "12-bands":
                    for k in models_dl[fold]:
                        print("Model:", k)
                        for m in ["no_classifier", "with_classifier"]:
                            print(m)
                            dir_clf = os.path.join(metrics_dir, str(fold), method, k, m)
                            y_pred = load_y(dir_clf)
                            y_pred_flat = np.vstack(y_pred).reshape(-1)
                            conf_mat = confusion_matrix(y_true_flat[idx], y_pred_flat[idx] > threshold)
                            print(conf_mat)

                            report += f"Model : {k}\n"
                            report += "With " if m == "with_classifier" else "No "
                            report += "classifier\n"
                            report += str(conf_mat) + "\n"
                            report += classification_report(y_true_flat[idx], y_pred_flat[idx] > threshold, target_names=["water","floating objects"]) + "\n"
            
                report += "\n---\n"
            
            with open(fold_report_path, "w") as f:
                f.write(report)
    


if __name__ == "__main__":
    main()