import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #모델 호출
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    #옵티마이져 실험세팅 준비
    #batch-size => batch_size 이런식으로 변환됨
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    #값 모아서 설정 
    args = parser.parse_args()

    #가속계산 gpu사용 가능한거 있는지 확인하고 있으면 사용 
    use_accel = not args.no_accel and torch.accelerator.is_available()

    #학습과정 랜덤 값의 시드값 설정하는 곳
    torch.manual_seed(args.seed)

    #use_accel에 따라 cpu 쓸지 gpu쓸지 결정
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    #값 추가 설정
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    
    # gpu사용 가능하면
    if use_accel:
        #값 추가 설정
        accel_kwargs = {'num_workers': 1, #worker(백그라운드 에서 미리 다음 데이터 준비해 주는거) 수
                        'persistent_workers': True, #한번 데이터를 볼때마다 worker를 종료하지 않고 대기 (속도가 빠름)
                       'pin_memory': True, #고정 메모리 사용 여부
                       'shuffle': True # 모델이 데이터 순서에 과적합 되는거를 방지하기 위해 데이터 섞음 여부
                       } 
        # 설정 주입
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        #값 추출
        transforms.ToTensor(),
        #값의 평균 값이랑 표준편차값 
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    #실 데이터 가져오고 없으면 다운로드
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    #테스트 데이터
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    #학습하기 편한 형식으로 변경
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #모델 선언 (device에 올림 gpu or cpu)
    model = Net().to(device)
    #optimizer(모델이 틀릴때 수정을 어떻게 할지 정하는것) 선언 
    #모델정보랑, 각 학습마다의 가중치를 설정
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    #가중치 관리 (나중에 가면 갈수록 섬세하기 다듬기 위해 가중치를 점점 줄여 나가는 부분)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        #모델 학습 
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()