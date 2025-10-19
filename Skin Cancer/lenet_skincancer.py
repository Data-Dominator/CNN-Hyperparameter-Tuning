import os, time, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
import optuna
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from deap import base, creator, tools, algorithms
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = "/kaggle/input/data-skin-cancer/Skin Cancer Dataset"
full_train = datasets.ImageFolder(root=os.path.join(data_dir, 'Train'), transform=transform)
full_test = datasets.ImageFolder(root=os.path.join(data_dir, 'Test'), transform=transform)

FIXED_BATCH_SIZE = 64

def train_and_evaluate(params, epochs=10, device=DEVICE, verbosity=False):
    lr, weight_decay, dropout_rate = params

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(full_train, batch_size=FIXED_BATCH_SIZE, shuffle=True, generator=g)
    test_loader = DataLoader(full_test, batch_size=FIXED_BATCH_SIZE, shuffle=False)

    model = LeNet(dropout_rate=dropout_rate)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(train_acc)

        model.eval()
        total_loss, correct, total = 0, 0, 0
        test_preds, test_probs, test_labels = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                test_preds.extend(predicted.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_losses.append(total_loss / len(test_loader))
        test_accuracies.append(correct / total)

    test_acc = accuracy_score(test_labels, test_preds)
    prw, recw, f1w, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted', zero_division=0)

    num_classes = 2
    test_labels_onehot = np.eye(num_classes)[test_labels]

    auc_dict = {}
    fpr_dict = {}
    tpr_dict = {}

    for i in range(num_classes):
        try:
            auc = roc_auc_score(test_labels_onehot[:, i], np.array(test_probs)[:, i])
            fpr, tpr, _ = roc_curve(test_labels_onehot[:, i], np.array(test_probs)[:, i])
        except ValueError:
            auc, fpr, tpr = 0.0, [0], [0]
        auc_dict[f"Class {i}"] = auc
        fpr_dict[f"Class {i}"] = fpr.tolist()
        tpr_dict[f"Class {i}"] = tpr.tolist()

    cmatrix = confusion_matrix(test_labels, test_preds)

    return dict(
        lr=lr, dropout_rate=dropout_rate, weight_decay=weight_decay,
        test_acc=test_acc,
        train_acc=train_accuracies[-1],
        precision_weighted=prw, recall_weighted=recw, f1_weighted=f1w,
        train_losses=train_losses, test_losses=test_losses,
        train_accuracies=train_accuracies, test_accuracies=test_accuracies,
        auc_dict=auc_dict, fpr_dict=fpr_dict, tpr_dict=tpr_dict,
        confusion_matrix=cmatrix
    )

def print_detailed_result(title, result, duration):
    print(f"\n{'='*10} {title} {'='*10}")
    print(f"Best Parameters:")
    print(f"  Learning Rate : {result['lr']:.2e}")
    print(f"  Dropout Rate  : {result['dropout_rate']:.2f}")
    print(f"  Weight Decay  : {result['weight_decay']:.2e}")
    print(f"Performance:")
    print(f"  Train Accuracy    : {result['train_acc']:.4f}")
    print(f"  Test Accuracy     : {result['test_acc']:.4f}")
    print(f"  Precision (Weighted): {result['precision_weighted']:.4f}")
    print(f"  Recall    (Weighted): {result['recall_weighted']:.4f}")
    print(f"  F1 Score  (Weighted): {result['f1_weighted']:.4f}")
    print(f"Time Taken: {duration:.2f} seconds")

    epochs = list(range(1, len(result["train_losses"]) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, result['train_accuracies'], label='Train Accuracy')
    plt.plot(epochs, result['test_accuracies'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, result['train_losses'], label='Train Loss')
    plt.plot(epochs, result['test_losses'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    for cls in result['auc_dict']:
        plt.plot(result['fpr_dict'][cls], result['tpr_dict'][cls], label=f'{cls} (AUC = {result["auc_dict"][cls]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for each Class')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant'])

    ax.invert_yaxis()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    print("\nEpoch-wise Numerical Data:")
    print("Epoch\tTrain_Acc\tTest_Acc\tTrain_Loss\tTest_Loss")
    for i, (tr_acc, ts_acc, tr_loss, ts_loss) in enumerate(zip(result['train_accuracies'], result['test_accuracies'], result['train_losses'], result['test_losses']), start=1):
        print(f"{i}\t{tr_acc:.4f}\t\t{ts_acc:.4f}\t\t{tr_loss:.4f}\t\t{ts_loss:.4f}")

    print("\nROC Curve Data for each Class:")
    for cls in result['auc_dict']:
        print(f"\nClass: {cls}")
        print("FPR\tTPR")
        for f, t in zip(result['fpr_dict'][cls], result['tpr_dict'][cls]):
            print(f"{f:.4f}\t{t:.4f}")

class LeNet(nn.Module):
    def __init__(self, dropout_rate):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

param_grid = {'lr':[1e-1,1e-2,1e-3,1e-4],'weight_decay':[0.0,1e-8,1e-4,1e-2],'dropout_rate':[0.1,0.2,0.3,0.4]}
print("Running Grid Search...")
start = time.time()
grid_results = [train_and_evaluate((p['lr'],p['weight_decay'],p['dropout_rate'])) for p in ParameterGrid(param_grid)]
grid_duration = time.time()-start
best_grid = max(grid_results, key=lambda x: x['test_acc'])
print_detailed_result("Grid Search", best_grid, grid_duration)

print("Running Random Search...")
random_trials = random.sample(list(ParameterGrid(param_grid)), 50)
start = time.time()
rand_results = [train_and_evaluate((p['lr'],p['weight_decay'],p['dropout_rate'])) for p in random_trials]
rand_duration = time.time()-start
best_rand = max(rand_results, key=lambda x: x['test_acc'])
print_detailed_result("Random Search", best_rand, rand_duration)

print("Running Optuna Search...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
opt_results = []
def optuna_objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    wd = trial.suggest_float('weight_decay', 0.0, 1e-2)
    do = trial.suggest_float('dropout_rate', 0.1, 0.4)
    metrics = train_and_evaluate((lr, wd, do))
    opt_results.append(metrics)
    return metrics['test_acc']

start = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(optuna_objective, n_trials=50, show_progress_bar=False)
opt_duration = time.time() - start
best_opt = max(opt_results, key=lambda x: x['test_acc'])
print_detailed_result("Optuna", best_opt, opt_duration)

print("Running Hyperopt...")
hyper_results = []
def hyper_objective(params):
    lr, wd, do = params['lr'], params['weight_decay'], params['dropout_rate']
    metrics = train_and_evaluate((lr, wd, do))
    hyper_results.append(metrics)
    return {'loss': -metrics['test_acc'], 'status': STATUS_OK}

hyper_space = {
    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-1)),
    'weight_decay': hp.uniform('weight_decay', 0.0, 1e-2),
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
}

start = time.time()
fmin(fn=hyper_objective, space=hyper_space, algo=tpe.suggest, max_evals=50, rstate=np.random.default_rng(SEED), show_progressbar=False)
hyper_duration = time.time() - start
best_hyper = max(hyper_results, key=lambda x: x['test_acc'])
print_detailed_result("Hyperopt", best_hyper, hyper_duration)

print("Running DEAP Algorithm...")
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_lr", lambda: 10**random.uniform(-4, -1))
toolbox.register("attr_wd", lambda: random.uniform(0.0, 1e-2))
toolbox.register("attr_do", lambda: random.uniform(0.1, 0.4))
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_lr, toolbox.attr_wd, toolbox.attr_do), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def deap_eval(ind):
    lr = max(1e-4, min(1e-1, ind[0]))
    wd = max(0.0, min(1e-2, ind[1]))
    do = max(0.1, min(0.4, ind[2]))
    ind[0], ind[1], ind[2] = lr, wd, do
    metrics = train_and_evaluate((lr, wd, do))
    ind.fitness.values = (metrics['test_acc'],)
    ind.metrics = metrics
    return ind.fitness.values

toolbox.register("evaluate", deap_eval)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

random.seed(SEED)
pop = toolbox.population(n=10)
start = time.time()
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=5, verbose=False)
ga_duration = time.time() - start
best_ind = tools.selBest(pop, 1)[0]
best_ga = best_ind.metrics
print_detailed_result("DEAP Algorithm", best_ga, ga_duration)

from scipy.stats import friedmanchisquare, f_oneway

grid_scores = [r['test_acc'] for r in grid_results]
rand_scores = [r['test_acc'] for r in rand_results]
optuna_scores = [r['test_acc'] for r in opt_results]
hyperopt_scores = [r['test_acc'] for r in hyper_results]
deap_scores = [ind.metrics['test_acc'] for ind in pop if hasattr(ind, 'metrics')]

min_len = min(len(grid_scores), len(rand_scores), len(optuna_scores), len(hyperopt_scores), len(deap_scores))
grid_scores = grid_scores[:min_len]
rand_scores = rand_scores[:min_len]
optuna_scores = optuna_scores[:min_len]
hyperopt_scores = hyperopt_scores[:min_len]
deap_scores = deap_scores[:min_len]

friedman_stat, friedman_p = friedmanchisquare(grid_scores, rand_scores, optuna_scores, hyperopt_scores, deap_scores)
print("\nFriedman Test Results:")
print(f"Statistic: {friedman_stat:.4f}, p-value: {friedman_p:.4f}")
if friedman_p < 0.05:
    print("=> Significant differences detected among models (p < 0.05)")
else:
    print("=> No significant differences (p ≥ 0.05)")

anova_stat, anova_p = f_oneway(grid_scores, rand_scores, optuna_scores, hyperopt_scores, deap_scores)
print("\nANOVA Results:")
print(f"Statistic: {anova_stat:.4f}, p-value: {anova_p:.4f}")
if anova_p < 0.05:
    print("=> Significant differences detected among models (p < 0.05)")
else:
    print("=> No significant differences (p ≥ 0.05)")

acc_dict = {
    'Grid Search': [res['test_acc'] for res in grid_results],
    'Random Search': [res['test_acc'] for res in rand_results],
    'Optuna': [res['test_acc'] for res in opt_results],
    'Hyperopt': [res['test_acc'] for res in hyper_results],
    'DEAP': [ind.metrics['test_acc'] for ind in pop if hasattr(ind, "metrics")]
}

import pandas as pd

data = []
for model_name, accs in acc_dict.items():
    for acc in accs:
        data.append({'Model': model_name, 'Accuracy': acc})

acc_df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='Accuracy', data=acc_df)
plt.title("Model Comparison")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.stats import ttest_rel, wilcoxon
from itertools import combinations

min_len = min(len(v) for v in acc_dict.values())
acc_dict_trimmed = {k: v[:min_len] for k, v in acc_dict.items()}

print("=== Pairwise Statistical Comparison ===")
print("{:<30} {:<15} {:<15} {:<15} {:<15}".format("Model Pair", "t-test p", "Wilcoxon p", "t-test sig", "Wilcoxon sig"))
print("=" * 95)

for model1, model2 in combinations(acc_dict_trimmed.keys(), 2):
    acc1 = acc_dict_trimmed[model1]
    acc2 = acc_dict_trimmed[model2]

    t_stat, p_ttest = ttest_rel(acc1, acc2)

    try:
        stat_wilcoxon, p_wilcoxon = wilcoxon(acc1, acc2)
    except ValueError:
        p_wilcoxon = float('nan')

    alpha = 0.05
    t_significant = p_ttest < alpha
    w_significant = p_wilcoxon < alpha

    print("{:<30} {:<15.4f} {:<15.4f} {:<15} {:<15}".format(
        f"{model1} vs {model2}",
        p_ttest,
        p_wilcoxon,
        "Yes" if t_significant else "No",
        "Yes" if w_significant else "No"
    ))

# Happy Coding!