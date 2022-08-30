from sched import scheduler
import torch 
import argparse
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import scatter 
import torch.nn.functional as F 
from model_cqa import get_basis, CQAFourier


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

'''
adding arguments here
'''

parser = argparse.ArgumentParser()
# default to be 12-spin lattice
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--J', type=list[float, float], default=[1.0, 0.5])
parser.add_argument('--irrep', type=list, default=[6,6])
parser.add_argument('--p', type=int, default=4)
parser.add_argument('--num_sites', type=int, default=12)
parser.add_argument('--numiter', type = int, default=500)
parser.add_argument('--lattice', type=str, default='rectangular')
parser.add_argument('--sample_size', type=int, default=200)
args = parser.parse_args()

def train(model, trial_state, optimizer, observable, device):
    model.train()                                                               
    # total_loss = 0                                                               
    optimizer.zero_grad()
    x = model(trial_state)
    observable = torch.view_as_complex(torch.stack([observable, torch.zeros_like(observable)], dim=-1))
    energy_expectation = torch.einsum('a, ab, b', torch.conj_physical(x),
                                                 observable, x)
    # print(energy_expectation)
    energy_expectation = torch.real(energy_expectation)
    energy_expectation.backward()
    optimizer.step()
    return energy_expectation.item()
        
def main():
    if args.lattice == 'kagome':
        lattice = [[(1,2), (1,5), (2,3), (3,4), (3,6),
             (5,7), (7, 9), (7,8), (6,10), (9,10), (10,11), (10,12)],
            [(2,5), (4,6), (6,9), (8,9), (11,12), (1,3),
            (2,4), (3,10), (2,6), (6,11), (6,12), (9,11),
            (9, 12), (5, 8), (1,7), (7, 10), (5,9)]]
    elif args.lattice =='rectangular':
        lattice =[[(1,2), (1,8), (2,3), (2,7), (3,4), (3,6), (4,5), (5,6), (5, 12),
            (6,7), (6, 11), (7, 10), (7, 8), (8,9), (9, 10), (10, 11), (11, 12)],

           [(1,3), (1,9), (1,7), (2,4), (2,8), (2, 10), (2, 6), (3, 11), (3,5), (3,7),
            (4, 12), (4, 6), (5, 7), (5, 11), (6, 8), (6, 10), (6, 12),  (7, 9), (7, 11), (8, 10),
            (9, 11), (10, 12)]]
    else: 
        raise NotImplementedError
    model = CQAFourier(args.J, args.num_sites, args.p, args.irrep, lattice, debug=False)
    optimizer = Adam(model.parameters(), lr=0.005)
    # optimizer = LBFGS(model.parameters(), lr=1)
    scheduler = ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=100, min_lr=0.00005)
    init_state = get_basis(model.dim, args.sample_size)
    observable = model.Heisenberg
    for i in range(args.numiter):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(model, init_state, optimizer, observable, args.device)
        scheduler.step(loss)
        print(f'Iteration: {i:03d}, LR: {lr:.5f}, Loss: {loss: .4f}')
        writer.add_scalar("Loss/train", loss, i)
    EDvalues, EDvectors = torch.linalg.eig(model.Heisenberg)
    EDvector = EDvectors[torch.argmin(torch.real(EDvalues))]
    EDvalue = EDvalues[torch.argmin(torch.real(EDvalues))]
    print('ED energy: {}'.format(EDvalue))
    writer.flush()
    writer.close()
if __name__ == "__main__":
    main()

