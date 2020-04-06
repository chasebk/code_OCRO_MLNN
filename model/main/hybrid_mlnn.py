#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:10, 06/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from mealpy.swarm_based import PSO, BFO
from mealpy.evolutionary_based import GA, DE, CRO
from model.root.hybrid.root_hybrid_mlnn import RootHybridMlnn


class GaMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = ga_paras["epoch"]
        self.pop_size = ga_paras["pop_size"]
        self.pc = ga_paras["pc"]
        self.pm = ga_paras["pm"]
        self.filename = "GA_MLNN-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = GA.BaseGA(self._objective_function__, self.problem_size, self.domain_range, self.print_train, self.epoch, self.pop_size, self.pc, self.pm)
        self.solution, self.best_fit, self.loss_train = md._train__()


class DeMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = de_paras["epoch"]
        self.pop_size = de_paras["pop_size"]
        self.wf = de_paras["wf"]
        self.cr = de_paras["cr"]
        self.filename = "DE_MLNN-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = DE.BaseDE(self._objective_function__, self.problem_size, self.domain_range, self.print_train, self.epoch, self.pop_size, self.wf, self.cr)
        self.solution, self.best_fit, self.loss_train = md._train__()


class PsoMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = pso_paras["epoch"]
        self.pop_size = pso_paras["pop_size"]
        self.c1 = pso_paras["c_minmax"][0]
        self.c2 = pso_paras["c_minmax"][1]
        self.w_min = pso_paras["w_minmax"][0]
        self.w_max = pso_paras["w_minmax"][1]
        self.filename = "PSO_MLNN-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = PSO.BasePSO(self._objective_function__, self.problem_size, self.domain_range, self.print_train,
                         self.epoch, self.pop_size, self.c1, self.c2, self.w_min, self.w_max)
        self.solution, self.best_fit, self.loss_train = md._train__()


class BfoMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, bfo_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.pop_size = bfo_paras["pop_size"]
        self.Ci = bfo_paras["Ci"]
        self.Ped = bfo_paras["Ped"]
        self.Ns = bfo_paras["Ns"]
        self.Ned = bfo_paras["Ned"]
        self.Nre = bfo_paras["Nre"]
        self.Nc = bfo_paras["Nc"]
        self.attract_repels = bfo_paras["attract_repels"]
        self.filename = "BFO_MLNN-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = BFO.BaseBFO(self._objective_function__, self.problem_size, self.domain_range, self.print_train,
                         self.pop_size, self.Ci, self.Ped, self.Ns, self.Ned, self.Nre, self.Nc, self.attract_repels)
        self.solution, self.best_fit, self.loss_train = md._train__()


class ABfoLSMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, abfols_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = abfols_paras["epoch"]
        self.pop_size = abfols_paras["pop_size"]
        self.Ci = abfols_paras["Ci"]
        self.Ped = abfols_paras["Ped"]
        self.Ns = abfols_paras["Ns"]
        self.N_minmax = abfols_paras["N_minmax"]
        self.filename = "ABFOLS_MLNN-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = BFO.ABFOLS(self._objective_function__, self.problem_size, self.domain_range, self.print_train,
                        self.epoch, self.pop_size, self.Ci, self.Ped, self.Ns, self.N_minmax)
        self.solution, self.best_fit, self.loss_train = md._train__()


class CroMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, cro_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = cro_paras["epoch"]
        self.pop_size = cro_paras["pop_size"]
        self.po = cro_paras["po"]
        self.Fb = cro_paras["Fb"]
        self.Fa = cro_paras["Fa"]
        self.Fd = cro_paras["Fd"]
        self.Pd = cro_paras["Pd"]
        self.G = cro_paras["G"]
        self.GCR = cro_paras["GCR"]
        self.k = cro_paras["k"]
        self.filename = "CRO_MLNN-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = CRO.BaseCRO(self._objective_function__, self.problem_size, self.domain_range, self.print_train,
                          self.epoch, self.pop_size, self.po, self.Fb, self.Fa, self.Fd, self.Pd, self.G, self.GCR, self.k)
        self.solution, self.best_fit, self.loss_train = md._train__()


class OCroMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ocro_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = ocro_paras["epoch"]
        self.pop_size = ocro_paras["pop_size"]
        self.po = ocro_paras["po"]
        self.Fb = ocro_paras["Fb"]
        self.Fa = ocro_paras["Fa"]
        self.Fd = ocro_paras["Fd"]
        self.Pd = ocro_paras["Pd"]
        self.G = ocro_paras["G"]
        self.GCR = ocro_paras["GCR"]
        self.k = ocro_paras["k"]
        self.restart_count = ocro_paras["restart_count"]
        self.filename = "OCRO_MLNN-sliding_{}-{}".format(root_base_paras["sliding"], root_hybrid_paras["paras_name"])

    def _training__(self):
        md = CRO.OCRO(self._objective_function__, self.problem_size, self.domain_range, self.print_train,
                      self.epoch, self.pop_size, self.po, self.Fb, self.Fa, self.Fd, self.Pd, self.G, self.GCR, self.k, self.restart_count)
        self.solution, self.best_fit, self.loss_train = md._train__()


