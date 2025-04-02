#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
Fourth moment localization

ref. JCP 137, 224114 (2012); DOI:10.1063/1.4769866
"""

import numpy
from functools import reduce, cached_property

from pyscf import lib
from pyscf.lib import logger
from pyscf.lo import boys
from pyscf import __config__


class FourthMoment(boys.OrbitalLocalizer):
    """The fourth moment localization optimizer that maximizes the orbital
    population

    Args:
        mol : Mole object

    Kwargs:
        mo_coeff : size (N,N) np.array
            The orbital space to localize for FM localization.
            When initializing the localization optimizer ``bopt = FM(mo_coeff)``,

            Note these orbitals ``mo_coeff`` may or may not be used as initial
            guess, depending on the attribute ``.init_guess`` . If ``.init_guess``
            is set to None, the ``mo_coeff`` will be used as initial guess. If
            ``.init_guess`` is 'atomic', a few atomic orbitals will be
            constructed inside the space of the input orbitals and the atomic
            orbitals will be used as initial guess.

            Note when calling .kernel(orb) method with a set of orbitals as
            argument, the orbitals will be used as initial guess regardless of
            the value of the attributes .mo_coeff and .init_guess.

    Attributes for FM class:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        conv_tol : float
            Converge threshold.  Default 1e-6
        conv_tol_grad : float
            Converge threshold for orbital rotation gradients.  Default 1e-3
        max_cycle : int
            The max. number of macro iterations. Default 100
        max_iters : int
            The max. number of iterations in each macro iteration. Default 20
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is preferred.
            Default 0.03.
        init_guess : str or None
            Initial guess for optimization. If set to None, orbitals defined
            by the attribute .mo_coeff will be used as initial guess. If set
            to 'atomic', atomic orbitals will be used as initial guess.
            Default 'atomic'
        exponent : int
            The power to define norm. It can be any integer >= 1. Default 2.

    Saved results

        mo_coeff : ndarray
            Localized orbitals

    """

    conv_tol = getattr(__config__, "lo_fourth_moment_FM_conv_tol", 1e-6)
    exponent = getattr(
        __config__, "lo_fourth_moment_FM_exponent", 2
    )  # any integer >= 1

    _keys = {"conv_tol", "exponent"}

    def __init__(self, mol, mo_coeff=None, mf=None):
        boys.OrbitalLocalizer.__init__(self, mol, mo_coeff)
        self._scf = mf

    def dump_flags(self, verbose=None):
        boys.OrbitalLocalizer.dump_flags(self, verbose)

    def gen_g_hop(self, u):
        mo_coeff = lib.dot(self.mo_coeff, u)

        # transform to mo basis
        r_ints = numpy.asarray(
            [reduce(lib.dot, (mo_coeff.conj().T, x, mo_coeff)) for x in self.r_ao_ints]
        )
        idxs1, idxs2 = numpy.triu_indices(3)
        rr_ints = numpy.empty(
            (3, 3, mo_coeff.shape[1], mo_coeff.shape[1]), dtype=numpy.float64
        )
        rr_ints[idxs1, idxs2] = rr_ints[idxs2, idxs1] = numpy.asarray(
            [
                reduce(
                    lib.dot, (mo_coeff.conj().T, self.rr_ao_ints[idx1, idx2], mo_coeff)
                )
                for idx1, idx2 in zip(idxs1, idxs2)
            ]
        )
        # (xxx + yyx + zzx, xxy + yyy + zzy, xxz + yyz + zzz)
        r3_ints = numpy.asarray(
            [reduce(lib.dot, (mo_coeff.conj().T, x, mo_coeff)) for x in self.r3_ao_ints]
        )
        # xxxx + xxyy + xxzz + yyxx + yyyy + zzzz + zzxx + zzyy + zzzz
        r4_ints = reduce(lib.dot, (mo_coeff.conj().T, self.r4_ao_ints, mo_coeff))

        # get diagonals, explicitly copy these to avoid jumps in memory
        r_ints_d = r_ints.diagonal(axis1=1, axis2=2).copy()
        rr_ints_d = rr_ints.diagonal(axis1=2, axis2=3).copy()
        r3_ints_d = r3_ints.diagonal(axis1=1, axis2=2).copy()
        r4_ints_d = r4_ints.diagonal().copy()
        xx_ints = numpy.moveaxis(rr_ints.diagonal(axis1=0, axis2=1), -1, 0)
        xx_ints_d = numpy.moveaxis(rr_ints_d.diagonal(axis1=0, axis2=1), -1, 0)

        # calculate orbital fourth moment
        if self.exponent > 1:
            fou_mom = (
                r4_ints_d
                - 4 * numpy.einsum("xi,xi->i", r3_ints_d, r_ints_d)
                + 2 * numpy.einsum("xi,yi,yi->i", xx_ints_d, r_ints_d, r_ints_d)
                + 4 * numpy.einsum("xyi,xi,yi->i", rr_ints_d, r_ints_d, r_ints_d)
                - 3 * numpy.einsum("xi,xi->i", r_ints_d, r_ints_d) ** 2
            )
            f0 = fou_mom ** (self.exponent - 1)
        else:
            f0 = numpy.ones(self.mo_coeff.shape[1], dtype=numpy.float64)

        # get diagonal intermediate involving three cartesian coordinates
        rrr_intermed = (
            -8
            * numpy.einsum(
                "xi,yi->yi", xx_ints_d, r_ints_d
            )  # missing for kappa @ f1 in hx in paper
            + 8 * r3_ints_d
            + 24 * numpy.einsum("xi,xi,yi->yi", r_ints_d, r_ints_d, r_ints_d)
            - 16 * numpy.einsum("xyi,xi->yi", rr_ints_d, r_ints_d)
        )

        # gradient
        g = numpy.zeros(2 * (self.mo_coeff.shape[1],), dtype=numpy.float64)
        g -= 2 * numpy.einsum("ij,j->ij", r4_ints, f0)  # additional factor 2
        g -= 8 * numpy.einsum("xyij,j,xj,yj->ij", rr_ints, f0, r_ints_d, r_ints_d)
        g += numpy.einsum("xij,j,xj->ij", r_ints, f0, rrr_intermed)
        g -= 4 * numpy.einsum("xij,j,yj,yj->ij", xx_ints, f0, r_ints_d, r_ints_d)
        g += 8 * numpy.einsum("xij,j,xj->ij", r3_ints, f0, r_ints_d)

        full_g = -self.exponent * (g - g.T)

        g = self.pack_uniq_var(full_g)

        # hessian diagonal
        h_diag = numpy.zeros(2 * (self.mo_coeff.shape[1],), dtype=numpy.float64)
        h_diag += 16 * numpy.einsum("xij,xij,j,yj->ij", r_ints, r_ints, f0, xx_ints_d)
        h_diag += 32 * numpy.einsum("xij,yij,j,yj->ij", xx_ints, r_ints, f0, r_ints_d)
        h_diag -= 32 * numpy.einsum("xij,xij,j->ij", r_ints, r3_ints, f0)
        h_diag += 64 * numpy.einsum("xij,xyij,j,yj->ij", r_ints, rr_ints, f0, r_ints_d)
        h_diag += 32 * numpy.einsum("xij,yij,j,xyj->ij", r_ints, r_ints, f0, rr_ints_d)
        h_diag -= 96 * numpy.einsum(
            "xij,yij,j,xj,yj->ij", r_ints, r_ints, f0, r_ints_d, r_ints_d
        )
        h_diag -= 48 * numpy.einsum(
            "xij,xij,j,yj,yj->ij", r_ints, r_ints, f0, r_ints_d, r_ints_d
        )

        r4_f0 = numpy.einsum("i,j->ij", r4_ints_d, f0)
        h_diag += 2 * (r4_f0 - numpy.einsum("ii->i...", r4_f0))
        xy_f0_x_y = numpy.einsum("xyi,j,xj,yj->ij", rr_ints_d, f0, r_ints_d, r_ints_d)
        h_diag += 8 * (xy_f0_x_y - numpy.einsum("ii->i...", xy_f0_x_y))
        r_f0_rrr_intermed = numpy.einsum("xi,j,xj->ij", r_ints_d, f0, rrr_intermed)
        h_diag -= r_f0_rrr_intermed - numpy.einsum("ii->i...", r_f0_rrr_intermed)
        xx_f0_y_y = numpy.einsum("xxi,j,yj,yj->ij", rr_ints_d, f0, r_ints_d, r_ints_d)
        h_diag += 4 * (xx_f0_y_y - numpy.einsum("ii->i...", xx_f0_y_y))
        r3_f0_r = numpy.einsum("xi,j,xj->ij", r3_ints_d, f0, r_ints_d)
        h_diag -= 8 * (r3_f0_r - numpy.einsum("ii->i...", r3_f0_r))

        if self.exponent > 1:
            kappa_f1 = numpy.zeros(2 * (self.mo_coeff.shape[1],), dtype=numpy.float64)
            kappa_f1 += 2 * r4_ints
            kappa_f1 += 8 * numpy.einsum("xi,yi,xyij->ij", r_ints_d, r_ints_d, rr_ints)
            kappa_f1 -= numpy.einsum("xi,xij->ij", rrr_intermed, r_ints)
            kappa_f1 -= 8 * numpy.einsum("xi,xij->ij", r_ints_d, r3_ints)
            kappa_f1 += 4 * numpy.einsum("xi,xi,yij->ij", r_ints_d, r_ints_d, xx_ints)

            if self.exponent > 2:
                h_diag += (
                    (self.exponent - 1)
                    * (fou_mom ** (self.exponent - 2))[:, numpy.newaxis].T
                    * kappa_f1.T**2
                )
            else:
                h_diag += (self.exponent - 1) * kappa_f1.T**2

        h_diag = self.exponent * self.pack_uniq_var(h_diag + h_diag.T)

        # get intermediates
        if self.exponent > 1:
            x_yy_ints_d_x_ints_d = numpy.einsum("yi,xi->xi", xx_ints_d, r_ints_d)
            x_x_ints_d_y_ints_d_y_ints_d = numpy.einsum("xi,yi,yi->xi", r_ints_d, r_ints_d, r_ints_d)
            x_xy_ints_d_y_ints_d = numpy.einsum("xyi,yi->xi", rr_ints_d, r_ints_d)
            interm1 = 8 * (
                x_yy_ints_d_x_ints_d 
                - r3_ints_d 
                - 3 * x_x_ints_d_y_ints_d_y_ints_d 
                + 2 * x_xy_ints_d_y_ints_d
            )

        x_f0_xxx_intermed = numpy.einsum("j,xj->xj", f0, rrr_intermed)
        x_yy_ints_f0_x_ints_d = numpy.einsum(
            "xij,j,yj->yij", 
            xx_ints, 
            f0, 
            r_ints_d, 
            optimize=['einsum_path', (1, 2), (0, 1)],
        )
        x_xy_ints_f0_y_ints_d = numpy.einsum(
            "xyij,j,xj->yij", 
            rr_ints, 
            f0, 
            r_ints_d, 
            optimize=['einsum_path', (1, 2), (0, 1)],
        )
        x_x3_ints_f0 = numpy.einsum("xij,j->xij", r3_ints, f0)
        x_x_ints_f0_yy_ints_d = numpy.einsum(
                "xij,j,yj->xij", 
                r_ints, 
                f0, 
                xx_ints_d, 
                optimize=['einsum_path', (1, 2), (0, 1)],
            )
        x_y_ints_f0_y_ints_d_x_ints_d = numpy.einsum(
                "xij,j,xj,yj->yij", 
                r_ints, 
                f0, 
                r_ints_d, 
                r_ints_d, 
                optimize=['einsum_path', (1, 3), (1, 2), (0, 1)]
            )
        x_y_ints_f0_xy_ints_d = numpy.einsum(
                "xij,j,xyj->yij", 
                r_ints, 
                f0, 
                rr_ints_d, 
                optimize=['einsum_path', (1, 2), (0, 1)]
            )
        x_x_ints_f0_y_ints_d_y_ints_d = numpy.einsum(
                "xij,j,yj,yj->xij", 
                r_ints, 
                f0, 
                r_ints_d, 
                r_ints_d,  
                optimize=['einsum_path', (2, 3), (1, 2), (0, 1)],
            )
        interm2 = 8 * (
            x_yy_ints_f0_x_ints_d
            + 2 * x_xy_ints_f0_y_ints_d 
            - x_x3_ints_f0
            + x_x_ints_f0_yy_ints_d # wrong prefactor in paper
            - 6 * x_y_ints_f0_y_ints_d_x_ints_d
            + 2 * x_y_ints_f0_xy_ints_d
            - 3 * x_x_ints_f0_y_ints_d_y_ints_d # missing in paper
        )

        x_x_int_d_x_int_d = 4 * numpy.einsum("yi,yi->i", r_ints_d, r_ints_d)
        xy_x_int_d_y_int_d = 8 * numpy.einsum("xi,yi->xyi", r_ints_d, r_ints_d)
        
        r_ints_f0_r_ints_d = 8 * numpy.einsum(
            "xij,j,xj->ij", 
            r_ints, 
            f0, 
            r_ints_d, 
            optimize=['einsum_path', (1, 2), (0, 1)],
        )
        f0_r_ints_d_r_ints_d = 4 * numpy.einsum(
            "j,yj,yj->j", 
            f0, 
            r_ints_d, 
            r_ints_d,
            optimize=['einsum_path', (1, 2), (0, 1)],
        )
        xy_x_ints_f0_y_ints_d = 16 * numpy.einsum(
            "xij,j,yj->xyij", 
            r_ints, 
            f0, 
            r_ints_d, 
            optimize=['einsum_path', (1, 2), (0, 1)],
        )
        xy_f0_x_ints_d_y_ints_d = 8 * numpy.einsum(
            "j,xj,yj->xyj",  
            f0, 
            r_ints_d, 
            r_ints_d,
            optimize=['einsum_path', (0, 1), (0, 1)],
        )

        x_f0_x_ints_d = 8 * numpy.einsum("j,xj->xj", f0, r_ints_d)
        x_x_ints_f0 = 8 * numpy.einsum("xij,j->xij", r_ints, f0)

        if self.exponent > 1:
            xy_ints_x_ints_d_y_ints_d = numpy.einsum(
                "xyij,xj,yj->ij",
                rr_ints,  
                r_ints_d, 
                r_ints_d, 
                optimize=['einsum_path', (1, 2), (0, 1)],
            )
            x_ints_xxx_intermed = numpy.einsum("xij,xj->ij", r_ints, rrr_intermed)
            xx_ints_y_ints_d_y_ints_d = numpy.einsum(
                "xij,yj,yj->ij", 
                xx_ints, 
                r_ints_d,
                r_ints_d, 
                optimize=['einsum_path', (1, 2), (0, 1)],
            )
            x3_ints_x_ints_d = numpy.einsum("xij,xj->ij", r3_ints, r_ints_d)
            interm3 = (
                2 * r4_ints
                + 8 * xy_ints_x_ints_d_y_ints_d
                - x_ints_xxx_intermed
                + 4 * xx_ints_y_ints_d_y_ints_d
                - 8 * x3_ints_x_ints_d
            )

        # hessian vector product
        def h_op(x):
            x = self.unpack_uniq_var(x)

            # terms containing f0
            # kappa @ x
            kappa_x = numpy.asarray([lib.dot(x, x_r_ints) for x_r_ints in r_ints])
            kappa_x_d = kappa_x.diagonal(axis1=1, axis2=2).copy()
            if self.exponent > 1:
                kappa_f1 = numpy.einsum("xi,xi->i", kappa_x_d, interm1)
            kappa_x += numpy.swapaxes(kappa_x, 1, 2)
            kappa_x_d *= 2
            hx = numpy.einsum("xij,xj->ij", kappa_x, x_f0_xxx_intermed)
            hx -= numpy.einsum("xij,xj->ij", interm2, kappa_x_d)

            # kappa @ xy
            kappa_xy = numpy.empty_like(rr_ints)
            kappa_xy[idxs1, idxs2] = kappa_xy[idxs2, idxs1] = numpy.asarray(
                [lib.dot(x, rr_ints[idx1, idx2]) for idx1, idx2 in zip(idxs1, idxs2)]
            )
            kappa_xy_d = kappa_xy.diagonal(axis1=2, axis2=3).copy()
            if self.exponent > 1:
                kappa_f1 += numpy.einsum("xxi,i->i", kappa_xy_d, x_x_int_d_x_int_d)
                kappa_f1 += numpy.einsum(
                    "xyi,xyi->i", kappa_xy_d, xy_x_int_d_y_int_d
                )
            kappa_xy += numpy.swapaxes(kappa_xy, 2, 3)
            kappa_xy_d *= 2
            hx -= numpy.einsum("ij,yyj->ij", r_ints_f0_r_ints_d, kappa_xy_d)
            hx -= numpy.einsum("xxij,j->ij", kappa_xy, f0_r_ints_d_r_ints_d)
            hx -= numpy.einsum("xyij,xyj->ij", xy_x_ints_f0_y_ints_d, kappa_xy_d)
            hx -= numpy.einsum("xyij,xyj->ij", kappa_xy, xy_f0_x_ints_d_y_ints_d)

            # kappa @ xxy
            kappa_xxy = numpy.asarray([lib.dot(x, x_r3_ints) for x_r3_ints in r3_ints])
            kappa_xxy_d = kappa_xxy.diagonal(axis1=1, axis2=2).copy()
            if self.exponent > 1:
                kappa_f1 -= 8 * numpy.einsum("xi,xi->i", kappa_xxy_d, r_ints_d)
            kappa_xxy += numpy.swapaxes(kappa_xxy, 1, 2)
            kappa_xxy_d *= 2
            hx += numpy.einsum("xij,xj->ij", kappa_xxy, x_f0_x_ints_d)
            hx += numpy.einsum("xij,xj->ij", x_x_ints_f0, kappa_xxy_d)

            # kappa @ xxyy
            kappa_xxyy = lib.dot(x, r4_ints)
            if self.exponent > 1:
                kappa_f1 += 2 * kappa_xxyy.diagonal()
            kappa_xxyy += kappa_xxyy.T
            hx -= 2 * numpy.einsum("ij,j->ij", kappa_xxyy, f0)

            # finish constructing kappa @ f1
            if self.exponent > 2:
                kappa_f1 *= (self.exponent - 1) * fou_mom ** (self.exponent - 2)

            # terms containing kappa @ f1
            if self.exponent > 1:
                hx -= numpy.einsum("ij,j->ij", interm3, kappa_f1)

            # multiply all terms by m
            hx *= self.exponent

            # term containing gradient
            hx += 0.5 * lib.dot(x, full_g)

            return self.pack_uniq_var(hx - hx.T)

        return g, h_op, h_diag

    def get_grad(self, u=None):
        if u is None:
            u = numpy.eye(self.mo_coeff.shape[1])
        mo_coeff = lib.dot(self.mo_coeff, u)

        # transform to mo basis
        r_ints = numpy.asarray(
            [reduce(lib.dot, (mo_coeff.conj().T, x, mo_coeff)) for x in self.r_ao_ints]
        )
        idxs1, idxs2 = numpy.triu_indices(3)
        rr_ints = numpy.empty(
            (3, 3, mo_coeff.shape[1], mo_coeff.shape[1]), dtype=numpy.float64
        )
        rr_ints[idxs1, idxs2] = rr_ints[idxs2, idxs1] = numpy.asarray(
            [
                reduce(
                    lib.dot, (mo_coeff.conj().T, self.rr_ao_ints[idx1, idx2], mo_coeff)
                )
                for idx1, idx2 in zip(idxs1, idxs2)
            ]
        )
        # (xxx + yyx + zzx, xxy + yyy + zzy, xxz + yyz + zzz)
        r3_ints = numpy.asarray(
            [reduce(lib.dot, (mo_coeff.conj().T, x, mo_coeff)) for x in self.r3_ao_ints]
        )
        # xxxx + xxyy + xxzz + yyxx + yyyy + zzzz + zzxx + zzyy + zzzz
        r4_ints = reduce(lib.dot, (mo_coeff.conj().T, self.r4_ao_ints, mo_coeff))

        # calculate orbital fourth moment
        if self.exponent > 1:
            fou_mom = (
                numpy.einsum("ii->i", r4_ints)
                - 4 * numpy.einsum("xii,xii->i", r3_ints, r_ints)
                + 2 * numpy.einsum("xxii,yii,yii->i", rr_ints, r_ints, r_ints)
                + 4 * numpy.einsum("xyii,xii,yii->i", rr_ints, r_ints, r_ints)
                - 3 * numpy.einsum("xii,xii->i", r_ints, r_ints) ** 2
            )
            f0 = fou_mom ** (self.exponent - 1)
        else:
            f0 = numpy.ones(self.mo_coeff.shape[1], dtype=numpy.float64)

        g = numpy.zeros(2 * (self.mo_coeff.shape[1],), dtype=numpy.float64)
        g -= 2 * numpy.einsum("ij,j->ij", r4_ints, f0)
        g -= 8 * numpy.einsum("xyij,j,xjj,yjj->ij", rr_ints, f0, r_ints, r_ints)
        g -= 8 * numpy.einsum("xij,j,yyjj,xjj->ij", r_ints, f0, rr_ints, r_ints)
        g += 8 * numpy.einsum("xij,j,xjj->ij", r_ints, f0, r3_ints)
        g += 24 * numpy.einsum(
            "xij,j,yjj,yjj,xjj->ij", r_ints, f0, r_ints, r_ints, r_ints
        )
        g -= 16 * numpy.einsum("xij,j,yjj,yxjj->ij", r_ints, f0, r_ints, rr_ints)
        g -= 4 * numpy.einsum("xxij,j,yjj,yjj->ij", rr_ints, f0, r_ints, r_ints)
        g += 8 * numpy.einsum("xij,j,xjj->ij", r3_ints, f0, r_ints)

        g = -self.exponent * self.pack_uniq_var(g - g.T)
        return g

    def cost_function(self, u=None):
        if u is None:
            u = numpy.eye(self.mo_coeff.shape[1])
        mo_coeff = lib.dot(self.mo_coeff, u)

        # transform to mo basis
        r_ints = lib.einsum(
            "ip,xpq,qi->xi", mo_coeff.conj().T, self.r_ao_ints, mo_coeff
        )
        rr_ints = lib.einsum(
            "ip,xypq,qi->xyi", mo_coeff.conj().T, self.rr_ao_ints, mo_coeff
        )
        # (xxx + yyx + zzx, xxy + yyy + zzy, xxz + yyz + zzz)
        r3_ints = lib.einsum(
            "ip,xpq,qi->xi", mo_coeff.conj().T, self.r3_ao_ints, mo_coeff
        )
        # xxxx + xxyy + xxzz + yyxx + yyyy + zzzz + zzxx + zzyy + zzzz
        r4_ints = lib.einsum(
            "ip,pq,qi->i", mo_coeff.conj().T, self.r4_ao_ints, mo_coeff
        )

        # calculate orbital fourth moment
        fou_mom = (
            r4_ints
            - 4 * numpy.einsum("xi,xi->i", r3_ints, r_ints)
            + 2 * numpy.einsum("xxi,yi,yi->i", rr_ints, r_ints, r_ints)
            + 4 * numpy.einsum("xyi,xi,yi->i", rr_ints, r_ints, r_ints)
            - 3 * numpy.einsum("xi,xi->i", r_ints, r_ints) ** 2
        )

        return numpy.sum(fou_mom**self.exponent)

    @cached_property
    def r_ao_ints(self):
        return self.mol.intor_symmetric("int1e_r")

    @cached_property
    def rr_ao_ints(self):
        return self.mol.intor_symmetric("int1e_rr").reshape(
            3, 3, self.mol.nao, self.mol.nao
        )

    @cached_property
    def r3_ao_ints(self):
        return numpy.einsum(
            "xxypq->ypq",
            self.mol.intor_symmetric("int1e_rrr").reshape(
                3, 3, 3, self.mol.nao, self.mol.nao
            ),
        )

    @cached_property
    def r4_ao_ints(self):
        return self.mol.intor_symmetric("int1e_r4")


FM = FourthMoment


if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = """
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    """
    mol.basis = "ccpvdz"
    mol.build()
    mf = scf.RHF(mol).run()

    mlo = FM(mol)
    mlo.verbose = 4
    mlo.exponent = 2  # integer >= 2
    mo0 = mf.mo_coeff[:, mf.mo_occ > 1e-6]
    mo = mlo.kernel(mo0)
