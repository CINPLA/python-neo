# -*- coding: utf-8 -*-
"""
Tests of io.exdirio
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.core import (Block, Segment, ChannelIndex, AnalogSignal,
                      Unit, SpikeTrain, Epoch)
from neo.test.iotest.common_io_test import BaseTestIO

try:
    import exdir
    HAVE_EXDIR = True
except ImportError:
    HAVE_EXDIR = False

from neo.io.exdirio import ExdirIO
import shutil
import numpy as np
import quantities as pq
import os

@unittest.skipUnless(HAVE_EXDIR, "Requires exdir")
class TestExdirIO(unittest.TestCase):

    def setUp(self):
        self.fname = '/tmp/test.exdir'
        if os.path.exists(self.fname):
            shutil.rmtree(self.fname)
        n_channels = 5
        n_samples = 20
        n_spikes = 50
        blk = Block()
        seg = Segment()
        blk.segments.append(seg)
        chx1 = ChannelIndex(index=np.arange(n_channels))
        chx2 = ChannelIndex(index=np.arange(n_channels)*2)
        blk.channel_indexes.extend([chx1, chx2])

        wf1 = np.random.random((n_spikes, n_channels, n_samples))
        ts1 = np.sort(np.random.random(n_spikes))
        t_stop1 = np.ceil(ts1[-1])
        sptr1 = SpikeTrain(times=ts1, units='s',
                           waveforms=np.random.random((n_spikes,
                                                       n_channels,
                                                       n_samples))*pq.V,
                           name='spikes 1',
                           description='sptr1',
                           t_stop=t_stop1, **{'id': 1})
        sptr1.channel_index = chx1
        unit1 = Unit(name='unit 1')
        unit1.spiketrains.append(sptr1)
        chx1.units.append(unit1)
        seg.spiketrains.append(sptr1)

        ts2 = np.sort(np.random.random(n_spikes))
        t_stop2 = np.ceil(ts2[-1])
        sptr2 = SpikeTrain(times=ts2, units='s',
                           waveforms=np.random.random((n_spikes,
                                                       n_channels,
                                                       n_samples))*pq.V,
                           description='sptr2',
                           name='spikes 2',
                           t_stop=t_stop2, **{'id': 2})
        sptr2.channel_index = chx2
        unit2 = Unit(name='unit 2')
        unit2.spiketrains.append(sptr2)
        chx2.units.append(unit2)
        seg.spiketrains.append(sptr2)

        wf3 = np.random.random((n_spikes, n_channels, n_samples))
        ts3 = np.sort(np.random.random(n_spikes))
        t_stop3 = np.ceil(ts3[-1])
        sptr3 = SpikeTrain(times=ts3, units='s',
                           waveforms=np.random.random((n_spikes,
                                                       n_channels,
                                                       n_samples))*pq.V,
                           description='sptr3',
                           name='spikes 3',
                           t_stop=t_stop3, **{'id': 3})
        sptr3.channel_index = chx2
        unit3 = Unit(name='unit 3')
        unit3.spiketrains.append(sptr3)
        chx2.units.append(unit3)
        seg.spiketrains.append(sptr3)

        t_stop = max([t_stop1, t_stop2, t_stop3]) * pq.s

        ana = AnalogSignal(np.random.random(n_samples),
                           sampling_rate=n_samples/t_stop,
                           units='V',
                           name='ana1')
        assert t_stop == ana.t_stop
        seg.analogsignals.append(ana)
        epo = Epoch(np.random.random(n_samples),
                    durations=[1]*n_samples*pq.s,
                    units='s',
                    name='epo1')
        seg.epochs.append(epo)
        self.blk = blk

    def test_write_block(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        # io.write_LFP(ana)

    def test_write_read_block(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        np.testing.assert_equal(len(blk.segments[0].spiketrains),
                                len(self.blk.segments[0].spiketrains))

    def test_write_read_block_sptr_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        sptrs = {sptr.annotations['id']: sptr
                 for sptr in self.blk.segments[0].spiketrains}
        sptrs_load = {sptr.annotations['id']: sptr
                      for sptr in blk.segments[0].spiketrains}
        for key in sptrs.keys():
            np.testing.assert_array_equal(sptrs[key], sptrs_load[key])
            np.testing.assert_equal(sptrs[key].name, sptrs_load[key].name)
            np.testing.assert_equal(sptrs[key].description, sptrs_load[key].description)
            for k, v in sptrs[key].annotations.items():
                if k == 'description' or k == 'name':
                    continue
                np.testing.assert_equal(v, sptrs_load[key].annotations[k])
            np.testing.assert_array_equal(sptrs[key].channel_index.index,
                                    sptrs_load[key].channel_index.index)

    def test_write_read_block_chxs_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        np.testing.assert_equal(len(self.blk.channel_indexes),
                                len(blk.channel_indexes))
        units = {unit.name: unit
                 for unit in self.blk.channel_indexes[0].units}
        units_load = {unit.name: unit
                      for unit in blk.channel_indexes[0].units}
        for key in units.keys():
            np.assert_equal(len(units[key].spiketrains[0]),
                            len(units_load[key].spiketrains[0]))
            sptr = units[key].spiketrains[0]
            sptr_load = units_load[key].spiketrains[0]
            np.testing.assert_array_equal(sptr, sptr_load)
            np.testing.assert_equal(units[key].name, units_load[key].name)

    def test_write_read_block_epo_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        epos = {epo.name: epo
                 for epo in self.blk.segments[0].epochs}
        epos_load = {epo.name: epo
                 for epo in blk.segments[0].epochs}
        for key in epos.keys():
            np.testing.assert_array_equal(epos[key], epos_load[key])
            np.testing.assert_array_equal(epos[key].durations, epos_load[key].durations)
            np.testing.assert_equal(epos[key].name, epos_load[key].name)

    def test_write_read_block_ana_equal(self):
        io = ExdirIO(self.fname)
        io.write_block(self.blk)
        exdir_dir = exdir.File(self.fname)
        lfp_group = exdir_dir['/processing/electrophysiology/channel_group_0']
        io.write_LFP(self.blk.segments[0].analogsignals, lfp_group)
        io = ExdirIO(self.fname)
        blk = io.read_block()
        anas = {ana.name: ana
                 for ana in self.blk.segments[0].analogsignals}
        anas_load = {ana.name: ana
                 for ana in blk.segments[0].analogsignals}
        for key in anas.keys():
            np.testing.assert_array_equal(anas[key], anas_load[key])
            np.testing.assert_equal(anas[key].name, anas_load[key].name)



if __name__ == "__main__":
    unittest.main()
