import gc
from tqdm.auto import tqdm

from utils import set_seeds

class EmbTrainer:
    def __init__(self, model, loaders, criterion, optim, scheduler,
                 verbose, device, random_seeds=42):
        self.model = model.to(device)
        self.loaders = loaders
        self.criterion = criterion.to(device)
        self.optim = optim
        self.scheduler = scheduler

        self.val_step = 100
        self.checkpoint_save_step = 100
        self.verbose = verbose
        self.device = device
        set_seeds(random_seeds)

    def loop(self, epochs_start, epochs_end):
        for epoch in tqdm(range(epochs_start, epochs_end)):
            self.logs = {}

            self.model.train()
            self.run_epoch('train', epoch)

        #         self.model.eval()
        #         if (epoch + 1) % self.val_step == 0:
        #             phase = 'val' if 'val' in self.loader else 'test'
        #             with torch.no_grad():
        #                 self.run_epoch(phase, epoch)

        #         if (epoch + 1) % self.checkpoint_save_step == 0:
        #             self.save_net(f'{self.base_path}/checkpoints/{self.date}_epoch_{epoch}.')

            if self.scheduler:
                self.scheduler.step()
        #     with torch.no_grad():
        #         self.run_epoch('test', epoch)
        gc.collect()

    def run_epoch(self, phase, epoch):
        """Run whole epoch."""
        running_acc = 0.0
        running_loss = 0.0
        for c_idx, o_idx, neg_idx_samples in self.loaders[phase]:
            c_idx, o_idx, neg_idx_samples = c_idx.to(self.device), o_idx.to(self.device), neg_idx_samples.to(
                self.device)
            c_emb = self.model(c_idx)
            o_emb = self.model(o_idx)
            b, k = len(neg_idx_samples), len(neg_idx_samples[0])
            neg_emb_samples = self.model(neg_idx_samples)  # .reshape(b, k, -1)
            loss = self.criterion(c_emb, o_emb, neg_emb_samples)
            if phase == 'train':
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            running_loss += loss.item() * b

        epoch_loss = running_loss / len(self.loaders[phase].dataset)

        self.logs[f'{phase}_log_loss'] = round(epoch_loss, 4)
        #         self.writer.add_scalar(f'Loss/{phase}', self.logs[f'{phase}_log_loss'], epoch + 1)
        if self.verbose:
            print(self.logs)
