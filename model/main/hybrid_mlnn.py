from model.optimizer.swarm import BFO, ABC, PSO, WOA
from model.optimizer.evolutionary import GA, DE, CRO
from model.optimizer.physics import QSO
from model.root.hybrid.root_hybrid_mlnn import RootHybridMlnn


class GaMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.ga_paras = ga_paras
        self.filename = "GA_MLNN-sliding_{}-nets_{}-ga_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], ga_paras)

    def _training__(self):
        ga = GA.BaseGA(root_algo_paras=self.root_algo_paras, ga_paras = self.ga_paras)
        self.solution, self.loss_train = ga._train__()



class DeMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.de_paras = de_paras
        self.filename = "DE_MLNN-sliding_{}-nets_{}-de_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], de_paras)

    def _training__(self):
        md = DE.BaseDE(root_algo_paras=self.root_algo_paras, de_paras = self.de_paras)
        self.solution, self.loss_train = md._train__()



class CroMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, cro_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.cro_paras = cro_paras
        self.filename = "CRO_MLNN-sliding_{}-nets_{}-cro_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], cro_paras)

    def _training__(self):
        cro = CRO.BaseCRO(root_algo_paras=self.root_algo_paras, cro_paras = self.cro_paras)
        self.solution, self.loss_train = cro._train__()



class OCroMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, cro_paras=None, ocro_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.cro_paras = cro_paras
        self.ocro_paras = ocro_paras
        self.filename = "OCRO_MLNN-sliding_{}-nets_{}-cro_para_{}-ocro_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], cro_paras, ocro_paras)

    def _training__(self):
        cro = CRO.OCRO(root_algo_paras=self.root_algo_paras, cro_paras = self.cro_paras, ocro_paras=self.ocro_paras)
        self.solution, self.loss_train = cro._train__()








class PsoMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.pso_paras = pso_paras
        self.filename = "PSO_MLNN-sliding_{}-nets_{}-pso_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], pso_paras)

    def _training__(self):
        pso = PSO.BasePSO(root_algo_paras=self.root_algo_paras, pso_paras = self.pso_paras)
        self.solution, self.loss_train = pso._train__()

class BfoMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, bfo_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.bfo_paras = bfo_paras
        self.filename = "BFO_MLNN-sliding_{}-nets_{}-bfo_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], bfo_paras)

    def _training__(self):
        md = BFO.BaseBFO(root_algo_paras=self.root_algo_paras, bfo_paras = self.bfo_paras)
        self.solution, self.loss_train = md._train__()



class ABfoLSMlnn(RootHybridMlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, abfols_paras=None):
        RootHybridMlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.abfols_paras = abfols_paras
        self.filename = "ABfoLS_MLNN-sliding_{}-nets_{}-abfols_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], abfols_paras)

    def _training__(self):
        md = BFO.ABFOLS(root_algo_paras=self.root_algo_paras, abfols_paras=self.abfols_paras)
        self.solution, self.loss_train = md._train__()


