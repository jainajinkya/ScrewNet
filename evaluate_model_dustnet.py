import argparse
import os

import torch

from dataset import ArticulationDataset
from metrics import angular_maad, maad, screw_loss
from models import ScrewNet
from utils import interpret_labels_ours

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test model for articulated object dataset."
    )
    parser.add_argument("--model-dir", type=str, default="models/")
    parser.add_argument("--model-name", type=str, default="test_lstm")
    parser.add_argument("--test-dir", type=str, default="../data/test/microwave/")
    parser.add_argument("--output-dir", type=str, default="./plots/")
    parser.add_argument(
        "--ntest",
        type=int,
        default=100,
        help="number of test samples (n_object_instants)",
    )
    parser.add_argument(
        "--ndof",
        type=int,
        default=1,
        help="how many degrees of freedom in the object class?",
    )
    parser.add_argument("--batch", type=int, default=40, help="batch size")
    parser.add_argument("--nwork", type=int, default=8, help="num_workers")
    parser.add_argument("--device", type=int, default=0, help="cuda device")
    parser.add_argument(
        "--dual-quat",
        action="store_true",
        default=False,
        help="Dual quaternion representation or not",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        help="screw, noLSTM, 2imgs, l2, baseline",
    )
    parser.add_argument(
        "--load-wts",
        action="store_true",
        default=False,
        help="Should load model wts from prior run?",
    )
    parser.add_argument("--obj", type=str, default="microwave")
    args = parser.parse_args()

    output_dir = os.path.join(
        os.path.abspath(args.output_dir), args.model_type, args.model_name
    )
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    # Plotting Histograms as percentages
    formatter = FuncFormatter(to_percent)
    global percent_scale

    print("Testing ScrewNet")
    best_model = ScrewNet(lstm_hidden_dim=1000, n_lstm_hidden_layers=1, n_output=8)
    test_set = ArticulationDataset(args.ntest, args.test_dir)

    best_model.load_state_dict(
        torch.load(os.path.join(args.model_dir, args.model_name + ".net"))
    )
    best_model.float().to(device)
    best_model.eval()

    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.nwork,
        pin_memory=True,
    )

    maad_l = torch.tensor([0], dtype=float, device=device)
    maad_m_ori = torch.tensor([0], dtype=float, device=device)
    maad_m_mag = torch.tensor([0], dtype=float, device=device)
    maad_th = torch.tensor([0], dtype=float, device=device)
    maad_d = torch.tensor([0], dtype=float, device=device)

    screw_ori = torch.tensor([0], dtype=float, device=device)
    screw_dist = torch.tensor([0], dtype=float, device=device)
    screw_th = torch.tensor([0], dtype=float, device=device)
    screw_d = torch.tensor([0], dtype=float, device=device)
    screw_ortho = torch.tensor([0], dtype=float, device=device)

    with torch.no_grad():
        for X in testloader:
            depth, labels = X["depth"].to(device), X["label"].to(device)
            pred = best_model(depth)

            pred = pred.view(pred.size(0), -1, 8)
            pred = pred[:, 1:, :]

            labels = interpret_labels_ours(
                labels, test_set.normalization_factor
            )  # Scaling m appropriately
            pred = interpret_labels_ours(pred, test_set.normalization_factor)

            # Calculate Error statistics
            batch_size = labels.size(0)
            maad_l += angular_maad(labels[:, :, :3], pred[:, :, :3]) * batch_size
            maad_m_ori += angular_maad(labels[:, :, 3:6], pred[:, :, 3:6]) * batch_size

            maad_m_mag += (
                maad(labels[:, :, 3:6].norm(dim=-1), pred[:, :, 3:6].norm(dim=-1))
                * batch_size
            )

            maad_th += maad(labels[:, :, -2], pred[:, :, -2]) * batch_size
            maad_d += maad(labels[:, :, -1], pred[:, :, -1]) * batch_size

            # Screw Loss
            ori, dist, th, d, ortho = screw_loss(target_=labels, pred_=pred)
            screw_ori += ori * batch_size
            screw_dist += dist * batch_size
            screw_th += th * batch_size
            screw_d += d * batch_size
            screw_ortho += ortho * batch_size

    # Report mean values
    maad_l /= test_set.length
    maad_m_ori /= test_set.length
    maad_m_mag /= test_set.length
    maad_th /= test_set.length
    maad_d /= test_set.length

    screw_ori /= test_set.length
    screw_dist /= test_set.length
    screw_th /= test_set.length
    screw_d /= test_set.length
    screw_ortho /= test_set.length

    print(
        "MAAD Losses:\nl_ori: {:.4f}, m_ori: {:.4f}, m_mag: {:.4f}, theta: {:.4f}, d: {:.4f}".format(
            maad_l.item(),
            maad_m_ori.item(),
            maad_m_mag.item(),
            maad_th.item(),
            maad_d.item(),
        )
    )

    print(
        "\nScrew Losses:\nOri: {:.4f}, Dist: {:.4f}, theta: {:.4f}, d: {:.4f}, Ortho: {:.4f}\n".format(
            screw_ori.item(),
            screw_dist.item(),
            screw_th.item(),
            screw_d.item(),
            screw_ortho.item(),
        )
    )
