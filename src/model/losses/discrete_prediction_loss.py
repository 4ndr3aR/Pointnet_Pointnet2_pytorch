import torch
import torch.nn as nn

from src.model.losses.utils import get_optimal_assignment, calculate_point_distance_loss, calculate_norm_distance_loss, calculate_angle_loss, \
    calculate_cost_matrix_normals
from src.utils.plane import SymPlane

def feature_transform_regularizer(trans=None):
    if trans is None:
        return 0.0
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def get_sde(points, pred_plane, true_plane, p=2):
    """
    :param points:
    :param pred_plane:
    :param true_plane:
    :param p:
    :return:
    """
    print(f'pred_plane\n{pred_plane}')
    print(f'true_plane\n{true_plane}')
    pred_plane = SymPlane.from_tensor(pred_plane)
    true_plane = SymPlane.from_tensor(true_plane, normalize=True)
    print(f'pred_plane\n{pred_plane}')
    print(f'true_plane\n{true_plane}')
    ppr = pred_plane.reflect_points(points)
    tpr = true_plane.reflect_points(points)
    print(f'{len(ppr) = }')
    print(f'{len(tpr) = }')
    print(f'ppr\n{ppr}')
    print(f'tpr\n{tpr}')
    loss = torch.norm(tpr - ppr, dim=0, p=p).mean()
    print(f'SDE loss: {loss}')
    return loss


def calculate_sde_loss(points, y_pred, y_true):
    """

    :param points:
    :param y_pred: M x 6
    :param y_true: M x 6
    :return:
    """
    m = y_pred.shape[0]
    loss = torch.tensor([0.0], device=y_pred.device)
    for i in range(m):
        loss += get_sde(points, y_pred[i], y_true[i])
    return loss / m


def calculate_loss_aux(
        points,
        y_pred,
        y_true,
        cost_matrix_method,
        weights,
        trans_feat = None,
        show_loss_log=False
):
    """
    :param show_loss_log:
    :param weights:
    :param cost_matrix_method:
    :param points: N x 3
    :param y_pred: M x 7
    :param y_true: K x 6
    :return:
    """

    print(f'points\n{points}')
    print(f'calculate_loss_aux() {y_pred.shape = }')
    print(f'y_pred\n{y_pred}')
    print(f'calculate_loss_aux() {y_true.shape = }')
    print(f'y_true\n{y_true}')
    print(f'weights\n{weights}')

    m = y_pred.shape[0]
    confidences = y_pred[:, -1]

    # c_hat : One-Hot M
    # matched_y_pred : K x 7
    print(f'{cost_matrix_method = }')
    c_hat, matched_y_pred = get_optimal_assignment(points, y_pred, y_true, cost_matrix_method)
    print(f'c_hat\n{c_hat}')
    print(f'matched_y_pred\n{matched_y_pred}')

    confidence_loss = nn.functional.binary_cross_entropy(confidences, c_hat.cuda()) * weights[0]

    sde_loss = calculate_sde_loss(points, matched_y_pred[:, 0:6], y_true) * weights[1]

    p_distance_loss = calculate_point_distance_loss(matched_y_pred[:, 0:6], y_true) * weights[2]
    n_distance_loss = calculate_norm_distance_loss (matched_y_pred[:, 0:6], y_true) * weights[2] / 10.

    angle_loss = calculate_angle_loss(matched_y_pred[:, 0:6], y_true) * weights[3]

    mat_diff_loss = feature_transform_regularizer(trans_feat)

    total_loss = confidence_loss + sde_loss + angle_loss + p_distance_loss + n_distance_loss + mat_diff_loss

    if show_loss_log:
        torch.set_printoptions  (linewidth=200)
        torch.set_printoptions  (precision=3)
        torch.set_printoptions  (sci_mode=False)
        print(f"Total_loss  : {total_loss.item():.2f}")
        print(f"Conf_loss   : {(confidence_loss / total_loss).item():.2f} | {confidence_loss.item()}")
        print(f"sde_loss    : {(sde_loss / total_loss).item():.2f} | {sde_loss.item()}")
        print(f"angle_loss  : {(angle_loss / total_loss).item():.2f} | {angle_loss.item()}")
        print(f"p_dist_loss : {(p_distance_loss / total_loss).item():.2f} | {p_distance_loss.item()}")
        print(f"n_dist_loss : {(n_distance_loss / total_loss).item():.2f} | {n_distance_loss.item()}")
        #print(f"mae_loss     : {(mae_loss / total_loss).item():.2f} | {mae_loss.item()}")

    return total_loss


def calculate_loss(
        batch,
        y_pred,
        cost_matrix_method=calculate_cost_matrix_normals,
        weights=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        trans_feat=None,
        show_losses=False,
):
    """
    :param batch: Tuple of idxs, points, sym_planes, transforms
        idxs : tensor of shape B
        points : tensor of shape B x N x 3
        y_true : List of B tensor of shape K x 6
        transforms : List of transforms used
    :param y_pred: tensor   B x H x 7
    :param weights:
    :param cost_matrix_method:
    :return:
    """
    print(f'BEGIN LOSS----------------------------------')
    print(f'--------------------------------------------')
    print(f'--------------------------------------------')
    _, points, y_true, _ = batch
    bs     = points.shape[0]
    loss   = torch.tensor([0.0], device=points.device)
    losses = torch.zeros(bs, device=points.device)

    if show_losses:
        torch.set_printoptions  (linewidth=200)
        torch.set_printoptions  (precision=3)
        torch.set_printoptions  (sci_mode=False)
        print(f"Points shape {points.shape}")
        print(f"Y_true shape {len(y_true)} - {y_true[0].shape = }")
        print(f"Y_pred shape {len(y_pred)} - {y_pred.shape = }")

    for b_idx in range(bs):
        print(f'[{b_idx}/{bs}] BEGIN LOSS FOR++++++++++++++++++++++++')
        print(f'++++++++++++++++++++++++++++++++++++++++++++')
        print(f'++++++++++++++++++++++++++++++++++++++++++++')
        curr_points = points[b_idx]
        curr_y_true = y_true[b_idx]
        curr_y_pred = y_pred[b_idx]
        print(f'{curr_y_true.shape = }')
        print(f'{curr_y_pred.shape = }')
        losses[b_idx] = calculate_loss_aux(
            curr_points, curr_y_pred, curr_y_true,
            cost_matrix_method, weights, trans_feat,
            show_losses)
        if show_losses:
            print(f"{[b_idx]} Points\n{curr_points}")
            print(f"{[b_idx]} Y_true\n{curr_y_true}")
            print(f"{[b_idx]} Y_pred\n{curr_y_pred}")
            print(f"{[b_idx]} Loss: {losses[b_idx].item()}")
        print(f'////////////////////////////////////////////')
        print(f'////////////////////////////////////////////')
        print(f'[{b_idx}/{bs}] END LOSS FOR//////////////////////////')
    #final_loss = loss / bs
    loss = torch.mean(losses)
    #loss = torch.sum(losses)
    if show_losses:
        print(f"Final loss: {loss.item()}")
        #print(f"Final loss: {final_loss.item()}")
    print(f'============================================')
    print(f'============================================')
    print(f'END LOSS====================================')
    return loss
    #return final_loss
