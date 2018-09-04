# ----------------------------------------------------------------------
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import math
import numpy

from nab.detectors.base import AnomalyDetector

class WindowedAverageDetector(AnomalyDetector):
  """ A sliding window detector that computes anomaly score of a data point
  by computing its average over a window
  of previous data points. The windowSize is tuned to give best performance
  on NAB.
  """

  def __init__(self, *args, **kwargs):
    super(WindowedAverageDetector, self).__init__(*args, **kwargs)

    self.windowSize = 100
    self.windowData = []
    self.stepBuffer = []
    self.stepSize = 50
    self.mean = 0
    self.std = 1


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore).
    The anomalyScore is the average over a sliding window of inputData values. 
    """

    anomalyScore = self.mean
    inputValue = inputData["value"]
    if len(self.windowData) > 0:
      anomalyScore = self.mean

    if len(self.windowData) < self.windowSize:
      self.windowData.append(inputValue)
      self._updateWindow()
    else:
      self.stepBuffer.append(inputValue)
      if len(self.stepBuffer) == self.stepSize:
        # slide window forward by stepSize
        self.windowData = self.windowData[self.stepSize:]
        self.windowData.extend(self.stepBuffer)
        # reset stepBuffer
        self.stepBuffer = []
        self._updateWindow()

    return (anomalyScore, )


  def _updateWindow(self):
    # caluclate mean for data in window
    self.mean = numpy.mean(self.windowData)
    # caluclate standard deviation for data in window
    self.std = numpy.std(self.windowData)
    if self.std == 0.0:
      self.std = 0.000001
