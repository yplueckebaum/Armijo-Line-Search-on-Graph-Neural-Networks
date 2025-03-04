# Line search project

## Datasets
- ✔ get ZINC dataset working
## GNN
- ✔ implement basic gnn
- ✔ incorporate line search into gnn
- ✔ update to proper to newest algorithm from phillips email
- ✔ try different c values 
- ✔ compares SalSA mit AdamSLS
  - ✔ fix how SalSA data is saved
  - ✔ Include in bar charts
- finalize traning script
- mv parameter (without first order momentum = AdamSLS?)

## Results
- train everything on ~100 epochs
- ✔ whats up with test loss? Try proper shuffle
- 🤔 test data makes sense? -> as far as i can tell yes it is just worse than default Adam
- table verglieich final val/c/optim/data etc.
  - What does that even mean? :D
  - Hab jz nen bar chart gemacht? vll table im report!
### Graphs
visualize:
- ✔ train/test loss
- ✔ learning rate adam vs adaptive (get from optimizer.state during runtime!)
- ✔ viz for different c values
- ✔ make log scale -> doesnt work
- ✔ include c value in legend
- ✔ consistant colors when selecting options
- ✔ Rangeslider
- ✔ ein graph pro val/train nicht zsm
  - Or grey out deselected lines maybe 
  - ✔  one graph compares "best c value" and the other one compares different c values?
  - ✔ But seperate train/test probably makes sense

### Report
- start writing report with chapters of what i did and why
### Notes

### future
- loss decrease SaLSA.py?
- interesannte sachen z.b. gradient norm in state packen
- sufficent decerase tracken, warum negativ?
- compare to Adam with lr scheduler
# def train(model, optimizer, features, adjacency_matrix, target):
    model.train()
    optimizer.zero_grad()

    # Encapsulate forward pass within a closure function
    closure = lambda: torch.mean((model(features, adjacency_matrix) - target) ** 2)

    # Use the optimizer's closure mechanism
    loss = optimizer.step(closure=closure)
    return loss.item()