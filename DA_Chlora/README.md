# Satellite Data Assimilation to forecast SSH

## TODO
- [x] Only 1 channel output, it is usedc to compute the gradient and both are used in the loss function in `gradient` mode
- [x] Remove the gradient calculation from the data loader, it is computed as the model trains
- [ ] Update the config file to add a `loss_mode` parameter to choose between `mse` and `mse_gradient`
- [ ] Update the README.md to reflect the new changes
