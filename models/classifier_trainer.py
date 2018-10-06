from tqdm import tqdm
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class MNISTTrainer:
    def __init__(self, model, train_loader, test_loader, lr, device, log_interval, n_epoch):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = device
        self.n_epoch = n_epoch
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.log_interval = log_interval
        self.summary_writer = SummaryWriter()
        self.global_training_step = 0

    def train_model(self):
        for epoch in tqdm(range(1, self.n_epoch + 1)):
            self._train(epoch)
            self._validate(epoch)

    def _train(self, epoch: int):
        print(f"start {epoch} epoch")

        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.global_training_step += 1

            if batch_idx % self.log_interval == 0:
                self.summary_writer.add_scalar('mnist/train_loss', loss.item(), self.global_training_step)

    def _validate(self, epoch):
        print(f"start {epoch} epoch")

        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.summary_writer.add_scalar('mnist/test_loss', test_loss, self.global_training_step)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        self.summary_writer.add_scalar('mnist/test_accuracy', accuracy, self.global_training_step)
