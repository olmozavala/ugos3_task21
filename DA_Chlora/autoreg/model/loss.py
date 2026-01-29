import torch
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def mae_loss(output, target):
    return F.l1_loss(output, target)

def mse_loss_with_mask(output, target, mask):
    return F.mse_loss(output * mask, target * mask)

def mae_loss_with_mask(output, target, mask):
    return F.l1_loss(output * mask, target * mask)

def gradient_loss(output, target, sobel_kernel_x, sobel_kernel_y, mask, valid_points, eps=1e-10):
    # Compute the gradients of the output and target
    grad_output_x = F.conv2d(output, sobel_kernel_x, padding=1)
    grad_output_y = F.conv2d(output, sobel_kernel_y, padding=1)
    grad_target_x = F.conv2d(target, sobel_kernel_x, padding=1)
    grad_target_y = F.conv2d(target, sobel_kernel_y, padding=1)
    # Compute the gradient magnitude
    grad_magnitude_output = torch.sqrt(grad_output_x**2 + grad_output_y**2 + eps)
    grad_magnitude_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + eps)
    # Normalize the gradients with mean 0 and std 1
    output_gradient = (grad_magnitude_output - grad_magnitude_output.mean()) / (grad_magnitude_output.std() + eps)
    target_gradient = (grad_magnitude_target - grad_magnitude_target.mean()) / (grad_magnitude_target.std() + eps)
    # Compute the loss
    return ((output_gradient - target_gradient)**2 * mask).sum() / valid_points

def curvature_loss(output, target, laplace_kernel, mask, valid_points, eps=1e-10):
    # Compute the curvature of the output and target
    curv_output = F.conv2d(output, laplace_kernel, padding=1)
    curv_target = F.conv2d(target, laplace_kernel, padding=1)
    # Compute the curvature magnitude
    curv_magnitude_output = torch.sqrt(curv_output**2 + eps)
    curv_magnitude_target = torch.sqrt(curv_target**2 + eps)
    # Normalize the curvatures with mean 0 and std 1
    output_curvature = (curv_magnitude_output - curv_magnitude_output.mean()) / (curv_magnitude_output.std() + eps)
    target_curvature = (curv_magnitude_target - curv_magnitude_target.mean()) / (curv_magnitude_target.std() + eps)
    # Compute the loss
    return ((output_curvature - target_curvature)**2 * mask).sum() / valid_points