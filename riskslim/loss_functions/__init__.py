import os
os.chdir('/mnt/phd/jihu/opioid/Code/riskslim/loss_functions')
from .log_loss import *
from .log_loss_weighted import *

try:
    from .fast_log_loss import *
except ImportError:
    print("warning: could not import fast log loss")
    print("warning: returning handle to standard loss functions")
    # todo replace with warning object
    import riskslim.loss_functions.log_loss as fast_log_loss

try:
    from .lookup_log_loss import *
except ImportError:
    print("warning: could not import lookup log loss")
    print("warning: returning handle to standard loss functions")
    # todo replace with warning object
    import riskslim.loss_functions.log_loss as lookup_log_loss

# Jingyuan
# import os
# # os.chdir('/Users/jingyuanhu/Desktop/Research/Interpretable Opioid/Code/risk-slim/riskslim/loss_functions')
# os.chdir('/mnt/phd/jihu/opioid/Code/riskslim/loss_functions')
# import log_loss
# import log_loss_weighted 

# try:
#     import fast_log_loss
# except ImportError:
#     print("warning: could not import fast log loss")
#     print("warning: returning handle to standard loss functions")
#     # todo replace with warning object
#     import log_loss as fast_log_loss

# try:
#     import lookup_log_loss
# except ImportError:
#     print("warning: could not import lookup log loss")
#     print("warning: returning handle to standard loss functions")
#     # todo replace with warning object
#     import log_loss as lookup_log_loss


