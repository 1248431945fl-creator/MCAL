import torch
from torch.utils.data import Dataset
from sklearn import metrics

class MultimodalDataset(Dataset):

    def __init__(self, brain_data, audio_data, labels):
        """
        Args:
            brain_data: numpy array, shape (n_samples, n_regions, n_regions)
            audio_data: numpy array, shape (n_samples, channels, length)
            labels: numpy array, shape (n_samples,)
        """
        self.brain_data = torch.FloatTensor(brain_data)
        self.audio_data = torch.FloatTensor(audio_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'brain': self.brain_data[idx],
            'audio': self.audio_data[idx],
            'label': self.labels[idx]
        }


def evaluate_model(mla, data_loader, mode='test'):
    mla.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in data_loader:
            brain_batch = batch_data['brain'].cuda()
            audio_batch = batch_data['audio'].cuda()
            label_batch = batch_data['label'].cuda()
            pred, *_ = mla(brain_batch, audio_batch, label_batch, mode=False)
            pred = torch.softmax(pred, dim=1)
            _, preds = torch.max(pred, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

    acc = metrics.accuracy_score(all_labels, all_preds)
    all_preds_clean = [int(x) for x in all_preds]
    all_labels_clean = [int(x) for x in all_labels]
    print("=" * 80)
    print(f"Predictions: {all_preds_clean}")
    print(f"Labels: {all_labels_clean}")
    print("=" * 80)
    if mode == 'test':
        precision = metrics.precision_score(all_labels, all_preds, zero_division=0)
        recall = metrics.recall_score(all_labels, all_preds)
        f1 = metrics.f1_score(all_labels, all_preds)
        return acc, precision, recall, f1, all_preds, all_labels

    return acc
