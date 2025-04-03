# Copyright: Johnson & Johnson Digital & Data Science (JJDDS)
#
# This file contains trade secrets of JJDDS. No part may be reproduced or transmitted in any
# form by any means or for any purpose without the express written permission of JJDDS.
#
# Purpose:  Polaris Application Source

import pandas as pd

def compute_penalizations(std, skip=False):

    if not skip:
        # Computes shortage and excess deviation penalizations, after validating Standard Cost values
        round_decimal_numbers = 2
        
        sdp = round(
            (sum(std) / len(std)) * 2, round_decimal_numbers
        )
        edp = round(
            sum(std) / len(std), round_decimal_numbers
        )

        # Computes item deviation penalization factor, which will be the Standard Cost for each item, divided by the mean
        idp = std/edp
    
    else:
        sdp = 0.0
        edp = 0.0
        idp = pd.Series([0.0], index=["No_Item"])

    return sdp, edp, idp