# python 3.7
"""Computes the editing directions regarding the region of interest."""

import os
import sys
import argparse
import signal
import numpy as np
from tqdm import tqdm

from RobustPCA import RobustPCA


def parse_args():
    """Parses arguments."""
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument('jaco_path', type=str,
                        help='Path to jacobian matrix.')
    parser.add_argument('--output_dir', type=str, default='outputs/directions_path',
                        help='Directory to save the results. If not specified,'
                             '`./outputs/directions` will be used by default.')
    parser.add_argument('--lamb', type=int, default=60,
                        help='The coefficient to control the sparsity')
    parser.add_argument('--max_iter', type=int, default=10000,
                        help='The max iteration for low-rank factorization')
    parser.add_argument('--num_relax', type=int, default=0,
                        help='Factor of relaxation for the non-zeros singular'
                             ' values')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    assert os.path.exists(args.jaco_path)
    Jacobians = np.load(args.jaco_path)
    image_size = Jacobians.shape[2]
    w_dim = Jacobians.shape[-1]
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    for ind in tqdm(range(Jacobians.shape[0])):
        Jacobian = Jacobians[ind]
        if len(Jacobian.shape) == 5:  # [2, H, W, 1, latent_dim]
            Jaco_fore = Jacobian[0, :, :, 0]
            Jaco_back = Jacobian[1, :, :, 0]
            Jaco_fore = np.reshape(Jaco_fore, [-1, w_dim])
            Jaco_back = np.reshape(Jaco_back, [-1, w_dim])
            coef_f = 1 / Jaco_fore.shape[0]
            coef_b = 1 / Jaco_back.shape[0]
            M_fore = coef_f * Jaco_fore.T.dot(Jaco_fore)
            B_back = coef_b * Jaco_back.T.dot(Jaco_back)
            # low-rank factorization on foreground
            RPCA = RobustPCA(M_fore, lamb=1/args.lamb)
            L_f, _ = RPCA.fit(max_iter=args.max_iter)
            rank_f = np.linalg.matrix_rank(L_f)
            # low-rank factorization on background
            RPCA = RobustPCA(B_back, lamb=1/args.lamb)
            L_b, _ = RPCA.fit(max_iter=args.max_iter)
            rank_b = np.linalg.matrix_rank(L_b)
            # SVD on the low-rank matrix
            _, _, VHf = np.linalg.svd(L_f)
            _, _, VHb = np.linalg.svd(L_b)
            F_principal = VHf[:rank_f]  # Principal space of foreground
            relax_subspace = min(max(1, rank_b - args.num_relax), w_dim-1)
            B_null = VHb[relax_subspace:].T  # Null space of background

            F_principal_proj = B_null.dot(B_null.T).dot(F_principal.T)  # Projection
            F_principal_proj = F_principal_proj.T
            F_principal_proj /= np.linalg.norm(
                F_principal_proj, axis=1, keepdims=True)

            np.save(save_dir + 'backward_direction.npy', F_principal_proj)
        
            Jaco_fore = Jacobian[1, :, :, 0]
            Jaco_back = Jacobian[0, :, :, 0]
            Jaco_fore = np.reshape(Jaco_fore, [-1, w_dim])
            Jaco_back = np.reshape(Jaco_back, [-1, w_dim])
            coef_f = 1 / Jaco_fore.shape[0]
            coef_b = 1 / Jaco_back.shape[0]
            M_fore = coef_f * Jaco_fore.T.dot(Jaco_fore)
            B_back = coef_b * Jaco_back.T.dot(Jaco_back)
            # low-rank factorization on foreground
            RPCA = RobustPCA(M_fore, lamb=1/args.lamb)
            L_f, _ = RPCA.fit(max_iter=args.max_iter)
            rank_f = np.linalg.matrix_rank(L_f)
            # low-rank factorization on background
            RPCA = RobustPCA(B_back, lamb=1/args.lamb)
            L_b, _ = RPCA.fit(max_iter=args.max_iter)
            rank_b = np.linalg.matrix_rank(L_b)
            # SVD on the low-rank matrix
            _, _, VHf = np.linalg.svd(L_f)
            _, _, VHb = np.linalg.svd(L_b)
            F_principal = VHf[:rank_f]  # Principal space of foreground
            relax_subspace = min(max(1, rank_b - args.num_relax), w_dim-1)
            B_null = VHb[relax_subspace:].T  # Null space of background

            F_principal_proj = B_null.dot(B_null.T).dot(F_principal.T)  # Projection
            F_principal_proj = F_principal_proj.T
            F_principal_proj /= np.linalg.norm(
                F_principal_proj, axis=1, keepdims=True)
            
            np.save(save_dir+'forward_direction.npy', F_principal_proj)

        else:
            raise ValueError(f'Shape of Jacobian is not correct!')
        


if __name__ == "__main__":
    main()
