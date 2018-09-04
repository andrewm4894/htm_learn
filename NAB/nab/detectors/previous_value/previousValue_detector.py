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

class PreviousValueDetector(AnomalyDetector):
  """ A sliding window detector that computes anomaly score of a data point
  by computing its average over a window
  of previous data points. The windowSize is 1, thus making it just the previous value.
  """

  def __init__(self, *args, **kwargs):
    super(PreviousValueDetector, self).__init__(*args, **kwargs)

    self.windowSize = 1
    self.windowData = []
    self.stepBuffer = []
    self.stepSize = 1
    self.mean = 0

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
