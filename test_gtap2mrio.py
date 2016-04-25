import unittest
import pandas as pd
import pandas.util.testing as pdt
import numpy as np
import gtap2mrio as g2m
import pdb
import os
import IPython


class TestG2M(unittest.TestCase):

    def  setUp(self):
        self.sysdir='./data_for_tests/'

    def assert_same_but_roworder(self, x, y, cols=None):

        if cols is not None:
            # In case of meaningless indexes, order relative to column and
            # reindex
            x = x.sort(x.columns[cols])
            x = x.reset_index(drop=True)
            y = y.sort(y.columns[cols])
            y = y.reset_index(drop=True)
        else:
            # Order columns alphabetically
            x = x.reindex_axis(sorted(x.columns), axis=1)
            y = y.reindex_axis(sorted(y.columns), axis=1)
            # Order rows alphabetically
            x = x.sort_index()
            y = y.sort_index()

        pdt.assert_frame_equal(x,y)



    def test_read_csvfile(self):

        ar = g2m.read_csvfile(os.path.join(self.sysdir,'vxmd_eu.csv'))
        at = pd.DataFrame.from_dict({
                 'northamer':
                    {'pdr': 0.77800000000000002, 'wht': 6.6100000000000003},
                 'eu': {'pdr': 172, 'wht': 5563}})

        self.assert_same_but_roworder(ar, at)

    def test_list_regions_and_sectors(self):
        r_regions, r_sectors = g2m.list_regions_and_sectors(self.sysdir)
        t_regions = ['northamer', 'eu']
        t_sectors = ['pdr', 'wht']
    #def test_read_gtap_data(self):
    #    parser = g2m.Gtap2Mrio(data_dir = self.sysdir)
    #    parser.read_gtap_data()

    #    vipm_t = pd.DataFrame.from_dict(
    #            {'eu': {'pdr': 0.25, 'wht': 2956.0},
    #             'northamer': {'pdr': 0.050999999999999997, 'wht': 0.495}})
    #    vigm_t = pd.DataFrame.from_dict(
    #            {'eu': {'pdr': 0.0, 'wht': 19.100000000000001},
    #             'northamer': {'pdr': 0.0, 'wht': 0.0050000000000000001}})

    #    self.assert_same_but_roworder(parser.vipm, vipm_t)
    #    self.assert_same_but_roworder(parser.vigm, vigm_t)

    #def test_organize_data(self):
    #    parser = g2m.Gtap2Mrio(data_dir = self.sysdir)
    #    parser.read_gtap_data()
    #    parser.organize_data()

    #    y_i_t = pd.DataFrame.from_dict(
    #            {'amount': {('northamer', 'pdr'): 0.050999999999999997,
    #                        ('northamer', 'wht'): 0.5,
    #                        ('eu', 'wht'): 2975.0999999999999,
    #                        ('eu', 'pdr'): 0.25}})

    #    self.assert_same_but_roworder(parser.y_i, y_i_t)


    def test_transform_3d_to_4d(self):
        df = pd.DataFrame.from_dict(
            {'input': {0: 'pdr', 1: 'wht', 2: 'pdr', 3: 'wht', 4: 'pdr',
                       5: 'wht', 6: 'pdr', 7: 'wht'},
             'amount': {0: 6.7199999999999998, 1: 0.023,
                        2: 0.099000000000000005, 3: 955.0,
                        4: 16.0, 5: 37.5, 6: 77.299999999999997, 7: 308.0},
             'region': {0: 'eu', 1: 'eu', 2: 'eu', 3: 'eu', 4: 'northamer',
                        5: 'northamer', 6: 'northamer', 7: 'northamer'},
             'output': {0: 'pdr', 1: 'pdr', 2: 'wht', 3: 'wht', 4: 'pdr',
                        5: 'pdr', 6: 'wht', 7: 'wht'}})

        colId = ['region', 'output']
        indexId=['region', 'input']

        a_r = g2m.transform_3d_to_4d(df,
                                      indexId=indexId,
                                      colId=colId ,
                                      valueId=['amount'])

        a_t = pd.DataFrame.from_dict(
              {('northamer', 'pdr'): {('northamer', 'pdr'): 16.0,
                                      ('northamer', 'wht'): 37.5,
                                      ('eu', 'wht'): 0.0,
                                      ('eu', 'pdr'): 0.0},
               ('northamer', 'wht'): {('northamer', 'pdr'): 77.299999999999997,
                                      ('northamer', 'wht'): 308.0,
                                      ('eu', 'wht'): 0.0,
                                      ('eu', 'pdr'): 0.0},
               ('eu', 'wht'): {('northamer', 'pdr'): 0.0,
                                      ('northamer', 'wht'): 0.0,
                                      ('eu', 'wht'): 955.0,
                                      ('eu', 'pdr'): 0.099000000000000005},
               ('eu', 'pdr'): {('northamer', 'pdr'): 0.0,
                                      ('northamer', 'wht'): 0.0,
                                      ('eu', 'wht'): 0.023,
                                      ('eu', 'pdr'): 6.7199999999999998}})
        a_t.columns.names=colId
        a_t.index.names=indexId

        self.assert_same_but_roworder(a_r, a_t)

    def test_read_3d_files(self):

        headers = ['destination', 'commodity', 'amount', 'origin']
        ar = g2m.read_3d_files('data_for_tests', 'vxmd_', '.csv', headers )
        at = pd.DataFrame.from_dict({'origin': {0: 'eu',
                                                1: 'eu',
                                                2: 'eu',
                                                3: 'eu',
                                                4: 'northamer',
                                                5: 'northamer',
                                                6: 'northamer',
                                                7: 'northamer'},
                                     'amount': {0: 0.77800000000000002,
                                                1: 6.6100000000000003,
                                                2: 172.0,
                                                3: 5563.0,
                                                4: 24.399999999999999,
                                                5: 481.0,
                                                6: 5.6100000000000003,
                                                7: 1148.0},
                                     'commodity': {0: 'pdr',
                                                   1: 'wht',
                                                   2: 'pdr',
                                                   3: 'wht',
                                                   4: 'pdr',
                                                   5: 'wht',
                                                   6: 'pdr',
                                                   7: 'wht'},
                                     'destination': {0: 'northamer',
                                                     1: 'northamer',
                                                     2: 'eu',
                                                     3: 'eu',
                                                     4: 'northamer',
                                                     5: 'northamer',
                                                     6: 'eu',
                                                     7: 'eu'}})

        self.assert_same_but_roworder(ar, at)

if __name__ == '__main__':
    unittest.main()

