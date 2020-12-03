"""Registry of agents as trainables for Tune."""


def _import_dyna_sac():
    from vmac.agents.dyna_sac import DynaSACTrainer

    return DynaSACTrainer


AGENTS = {
    "Dyna-SAC": _import_dyna_sac,
}
