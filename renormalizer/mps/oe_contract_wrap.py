# wraps opt_einsum contraction to show memory errors
import logging

import opt_einsum as oe

from renormalizer.mps.backend import MEMORY_ERRORS, ARRAY_TYPES, xp


logger = logging.getLogger(__name__)


def log_error(e, args, kwargs):
    logger.exception(e)
    logger.fatal("The arguments are:")
    for i, arg in enumerate(args):
        if isinstance(arg, ARRAY_TYPES):
            logger.fatal(f"{i} Array type: {type(arg)}, shape:{arg.shape}")
        else:
            logger.fatal(f"{i} Non-array argument: {arg}")
    for k, v in kwargs.items():
        logger.fatal(f"{k}: {v}")


def update_kwargs(args, kwargs):
    # in Reno, the expressions are usually simple (at most 10 tensors involved)
    # Yet the performance requirement is critical - we frequently hit the memory limit
    if "optimize" not in kwargs:
        if len(args) <= 15:
            algo = "optimal"
        elif len(args) <= 25:
            algo = "dp"
        else:
            algo = "auto-hq"
        # modify in-place
        kwargs["optimize"] = algo

def oe_contract(*args, **kwargs):
    update_kwargs(args, kwargs)
    try:
        return oe.contract(*args, **kwargs)
    except MEMORY_ERRORS as e:
        logger.fatal("Out of memory error calling oe.contract")
        log_error(e, args, kwargs)
        raise e


def oe_contract_expression(*args, **kwargs):
    update_kwargs(args, kwargs)
    expr = oe.contract_expression(*args, **kwargs)
    def expr_wrapped(matrix: xp.ndarray, *args2, **kwargs2):
        try:
            return expr(matrix, *args2, **kwargs2)
        except MEMORY_ERRORS as e:
            logger.fatal("Out of memory error calling oe contract expression")
            log_error(e, args, kwargs)
            logger.fatal(f"Input matrix type: {type(matrix)}, shape: {matrix.shape}")
            raise e
    return expr_wrapped
