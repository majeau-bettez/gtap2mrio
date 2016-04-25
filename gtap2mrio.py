""" Class to generate an MRIO from GTAP variables.

Based on :
    Peters, Glen Philip, Robbie Andrew, and James Lennox. 2011. “CONSTRUCTING
    AN ENVIRONMENTALLY-EXTENDED MULTI-REGIONAL INPUT–OUTPUT TABLE USING THE
    GTAP DATABASE.” Economic Systems Research 23 (2): 131–52.
    doi:10.1080/09535314.2011.563234.
"""

import pandas as pd
import numpy as np
import re
import os
import glob
import scipy.io
import IPython

class Gtap2Mrio(object):

    def __init__(self, data_dir=None):

        # directories
        self.data_dir = data_dir

        # Dimensions
        self.regions=None
        self.sectors=None

        # GTAP matrices
        self.vdfm = None
        self.vifm = None
        self.vxmd = None
        self.vipm = None
        self.vigm = None

        # Environmental extensions
        self.__F = None
        self.__S = None

    """ DATA INPUT """

    def read_gtap_data(self):

        # Read final demands by households and gov. (domestic and imports)
        self.vipm = read_csvfile(os.path.join(self.data_dir, 'vipm.csv'))
        self.vdpm = read_csvfile(os.path.join(self.data_dir, 'vdpm.csv'))
        self.vigm = read_csvfile(os.path.join(self.data_dir, 'vigm.csv'))
        self.vdgm = read_csvfile(os.path.join(self.data_dir, 'vdgm.csv'))

        # Read sector exports
        headers= ['destination', 'commodity', 'amount', 'origin']
        self.vxmd = read_3d_files(self.data_dir, 'vxmd_', '.csv', headers)

        # Get dimensions
        self.sectors = self.vxmd['commodity'].unique().tolist()
        self.regions = self.vxmd['origin'].unique().tolist()

        def separate_capital(df, the_label='cgds', the_column='output',
                             pivot_headers=['input', 'region', 'amount'],
                             toreindex=None):

            # filter for capital flows, based on value (the_label) in a column
            # (the_column)
            bo_capital = df[the_column] == the_label

            # all other columns
            remaining_columns = [i for i in df.columns if i != the_column]

            # separate capital
            capital = df.ix[bo_capital, remaining_columns]
            df = df.drop(df.ix[bo_capital,:].index)

            # reformat
            capital = capital.pivot(pivot_headers[0],
                                    pivot_headers[1],
                                    pivot_headers[2])
            capital.index.name=''
            capital.columns.name=''

            if toreindex is not None:
                capital = capital.reindex_like(toreindex)

            return df, capital

        # Read domestic exchanges
        headers=['output', 'input', 'amount', 'region']
        dom = read_3d_files(self.data_dir, 'vdfm_', '.csv', headers)
        self.vdfm, self.vdkm = separate_capital(dom, toreindex=self.vdpm)

        # Read sector imports
        headers= ['output', 'input', 'amount', 'region']
        imports = read_3d_files(self.data_dir, 'vifm_', '.csv', headers)
        self.vifm, self.vikm = separate_capital(imports, toreindex=self.vipm)


        # Read total production
        vom = read_csvfile(os.path.join(self.data_dir, 'vom+.csv'))
        self.vom = vom.ix[self.sectors, :]



    """ Intermediate Variables """

    @property
    def y_m(self):
        return self._aggregate_final_demands([self.vipm, self.vigm, self.vikm])

    @property
    def vim_iS(self):
        """ L'import total de la commodité i dans la région s, tel que calculé
        par l'équation 2 dans Peters et al (2011)."""

        # import by facilities
        vifm_iS = self.vifm.groupby(['region','input'] ).sum().unstack(0)
        vifm_iS.columns = vifm_iS.columns.droplevel()

        # added to imports by final consumers
        return vifm_iS + self.vipm + self.vigm + self.vikm

    @property
    def e_iRS(self):
        e = pd.pivot_table(self.vxmd,
                                 index=['origin','commodity'],
                                 columns=['destination'])
        e.columns = e.columns.droplevel(0)
        e.index.names = ['region','input']
        e.columns.names = ['region']
        return self.reindex(e, axes=(0,))

    @property
    def imports(self): # TODO find better name
        a = pd.pivot_table(self.vifm,
                           index=['input'],
                           columns=['region', 'output'])
        a.columns = a.columns.droplevel(0)
        return a


    @property
    def Z_rr(self):
        return self.reindex(transform_3d_to_4d(self.vdfm,
                                                 indexId=['region', 'input'],
                                                 colId = ['region', 'output'],
                                                 valueId=['amount']))
    @property
    def y_rr(self):
        return self._aggregate_final_demands([self.vdpm, self.vdgm, self.vdkm])


    @property
    def Z_rs(self):
        # calculat norm imports
        norm_imports = self.imports.copy()
        for region in self.regions:
            norm_imports[region] = (self.imports.ix[:,region].T/
                                         self.vim_iS.ix[:,region]).T
        norm_imports = norm_imports.fillna(0.0)
        # Calculate Z_rs
        Z_rs = self.e_iRS * norm_imports
        return self.reindex(Z_rs)

    @property
    def y_rs(self):
        # normalize imports
        ratio = self.y_m/vector_from_2d(self.vim_iS, indexlike=self.y_rr.index)

        # Calculate y_rs
        y_rs = self.e_iRS.sum(1).to_frame('amount') * ratio
        y_rs = self.reindex(y_rs, axes=(0,))
        return y_rs.fillna(0.0)


    """ FINAL VARIABLES """

    @property
    def y(self):
        return self.y_rr + self.y_rs

    @property
    def Z(self):
        return self.Z_rr + self.Z_rs

    @property
    def x(self):
        return self.reindex(vector_from_2d(self.vom), axes=(0,))

    @property
    def A(self):
        A = self.Z / self.x.sum(1)
        return A.fillna(0.0)


    """ FINAL VARIABLES -- EXTENSTIONS """

    @property
    def S(self):
        if self.__S is not None:
            return self.__S
        else:
            return self.__F / self.x.sum(1)
    @S.setter
    def S(self, S):

        # Assign variable
        self.__S = S

        # prevent conflicting data on F and S
        if S is not None:
            self.__F = None

    @property
    def F(self):
        if self.__F is not None:
            return self.__F
        else:
            return self.__S * self.x.sum(1)

    @F.setter
    def F(self, F):
        self.__F = F
        if F is not None:
            self.__S = None


    """ PROCESSING METHODS """

    def _aggregate_final_demands(self, the_final_demands):
        # Calculate import by final users
        for i in the_final_demands:
            try:
                y = y + vector_from_2d(i)
            except NameError:
                y = vector_from_2d(i)
        return self.reindex(y, axes=(0,))


    def reindex(self, a, axes=(0,1)):
        for axis in axes:
            a = a.reindex_axis(self.regions, level=0, axis=axis)
            a = a.reindex_axis(self.sectors, level=1, axis=axis)
        return a


    """ EXPORT METHODS """

    def savemat_coefficients(self, filename):
        matdict = {'A': self.A.values,
                   'S':self.S.values,
                   'y':self.y.values,
                   'PRO': ['/ '.join(i).strip() for i in self.A.index.values],
                   'STR': ['/ '.join(i).strip() for i in self.S.columns.values]
                   }
        scipy.io.matlab.savemat(filename, matdict)

    """ POTENTIAL QUALITY CHECKS TO BE IMPLEMENTED """
    # Check whether imports and exports are sort of balanced.
    # Do not expect exact match: the relative difference between exports and imports of products: margin M
    # (self.e_iRS.sum(1).groupby(level=1).sum() - self.imports.sum(1).reindex(self.sectors))/self.imports.sum(1).reindex(self.sectors)    

    ## Relative difference between x and Ze + y, i.e., the transport margin (see Peters et al.)
    #diff = (self.x - (self.Z.sum(1).to_frame(name='amount') + self.y_rr + self.y_rs))
    #reldiff = diff/self.x
    #reldiff_tot = diff.sum()/self.x.sum()
    #reldiff_tot
def read_csvfile(path):

    # Initial read
    a = pd.DataFrame.from_csv(path)

    # Drop total column
    for i in [0, 1]:
        try:
            a = a.drop('Total', i)
        except ValueError:
            pass
    # Drop empty columns, followed by empty rows
    a = a.dropna(1).dropna(0)

    # Rename indexes and columns
    a.index.name = ''
    a.columns.name = ''

    # Clean up index and column labels
    def cleanup(alist):
        clean = [re.sub('^[0-9]+ +','', k) for k in alist]
        return [k.lower() for k in clean]
    a.columns = cleanup(a.columns.tolist())
    a.index = cleanup(a.index.tolist())
    return a

def read_3d_files(path, name_start, name_end, headers=None):

    fullstart = os.path.join(path, name_start)
    b = None
    for filename in glob.iglob(fullstart + '*' + name_end):

        # Get index of the third dimension from filename
        var_name = re.sub('^' + fullstart, '', filename)
        var_name = re.sub(name_end+'$', '', var_name)

        # read and clean CSV
        a = read_csvfile(filename)

        # unpivot, and add third dimension
        a = a.unstack()
        a = a.reset_index()
        if headers:
            a[headers[-1]] = var_name

        # Append to previous file table
        try:
            b = pd.concat((b, a))
            b = b.reset_index(drop=True)
        except:
            b = a.copy()

    # Finalize headers and return
    if (b is not None) and headers:
        b.columns=headers
    return b

def transform_3d_to_4d(data, indexId, colId, valueId):
    df = data.copy()
    redundantId = list(set(indexId).intersection(colId))[0]
    colIdtmp = [k.replace(redundantId, 'tmp') for k in colId]

    df['tmp'] = df[redundantId]
    a = pd.pivot_table(df,  index=indexId, columns=colIdtmp, values=valueId)
    a.columns = a.columns.droplevel(0)
    a.columns.names = colId
    return a.fillna(0.0)

def vector_from_2d(data, index_name='sector', columns_name='region',
        index_cols=['region','sector'], value_col='amount',
        indexlike=None):
    a = data.copy()
    a.columns.name = columns_name
    a.index.name = index_name
    # unpivot
    a = a.stack()

    # order index levels
    a = a.reset_index().set_index(index_cols)
    a.columns = [value_col]
    
    # reindex to match
    if indexlike is not None:
        a = a.reindex_axis(indexlike, axis=0)
    return a

def list_regions_and_sectors(path):

    files = glob.glob(os.path.join(path, 'vxmd*'))
    d = read_csvfile(files[1])
    regions = d.columns.tolist()
    sectors = d.index.tolist()
    return regions, sectors
def one_over(x):
    """Simple function to invert each element of vector. if 0, stays 0, not Inf

    * Element-wise divide a vector of ones by x
    * Replace any instance of Infinity by 0

    Parameters
    ----------
    x : vector to be diagonalized

    Returns
    --------
    y : inversed vector
       Values = 1/coefficient, or 0 if coefficient == 0

    """
    y = 1 / x
    y[y == np.Inf] = 0
    return y

