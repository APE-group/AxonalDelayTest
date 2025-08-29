#!/usr/bin/env python
# coding: utf-8
#  compare_sim_prediction_lib.py
#  Copyright © 2025   Pier Stanislao Paolucci   <pier.paolucci@roma1.infn.it>
#  Copyright © 2025   Elena Pastorelli          <elena.pastorelli@roma1.infn.it>
#
#  SPDX-License-Identifier: GPL-3.0-only
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import matplotlib as plt
import pandas as pd

def compare_csv_files(csv_file_1, csv_file_2, threshold=1e-8):
    """
    Compare 'syn_ID', 'start_syn_value', 'final_syn_value' row-by-row
    with a relative difference threshold. Return True if all OK, else False.
    """
    df1 = pd.read_csv(csv_file_1).sort_values(by='syn_ID').reset_index(drop=True)
    df2 = pd.read_csv(csv_file_2).sort_values(by='syn_ID').reset_index(drop=True)

    needed_cols = ['syn_ID', 'start_syn_value', 'final_syn_value']
    for col in needed_cols:
        if col not in df1.columns or col not in df2.columns:
            print(f"Missing column '{col}' in one of the CSVs.")
            return False

    if len(df1) != len(df2):
        print("Mismatch in row counts.")
        return False

    returnFalse = False
    totalMismatch = 0
    failureList = []
    for i in range(len(df1)):
        syn_id_1 = df1.loc[i, 'syn_ID']
        syn_id_2 = df2.loc[i, 'syn_ID']
        if syn_id_1 != syn_id_2:
            print(f"Row {i} mismatch in syn_ID: {syn_id_1} vs {syn_id_2}")
            return False

        
        for col in ['start_syn_value', 'final_syn_value']:
            val1 = df1.loc[i, col]
            val2 = df2.loc[i, col]
            denom = max(1e-15, abs(val1), abs(val2))
            rel_diff = abs(val1 - val2)/denom
            if rel_diff > threshold:
                print(f"Row {i}, syn_ID={syn_id_1}, col={col} mismatch: {val1} vs {val2}, rel_diff={rel_diff}")
                totalMismatch += 1
                returnFalse = True
                failureList.append(i)
    if returnFalse:
        print(f"Files {csv_file_1} and {csv_file_2} mismatch")
        print("Total number of mismatches: ", totalMismatch)
        print("Failure list:", failureList)
        return False, failureList

    print(f"Files {csv_file_1} and {csv_file_2} match within threshold={threshold}")
    return True, failureList
