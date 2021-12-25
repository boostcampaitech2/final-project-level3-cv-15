# -*- coding: utf-8 -*- #
# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions resource transforms and symbols dict.

A resource transform function converts a JSON-serializable resource to a string
value. This module contains built-in transform functions that may be used in
resource projection and filter expressions.

NOTICE: Each TransformFoo() method is the implementation of a foo() transform
function. Even though the implementation here is in Python the usage in resource
projection and filter expressions is language agnostic. This affects the
Pythonicness of the Transform*() methods:
  (1) The docstrings are used to generate external user documentation.
  (2) The method prototypes are included in the documentation. In particular the
      prototype formal parameter names are stylized for the documentation.
  (3) The 'r', 'kwargs', and 'projection' args are not included in the external
      documentation. Docstring descriptions, other than the Args: line for the
      arg itself, should not mention these args. Assume the reader knows the
      specific item the transform is being applied to. When in doubt refer to
      the output of $ gcloud topic projections.
  (4) The types of some args, like r, are not fixed until runtime. Other args
      may have either a base type value or string representation of that type.
      It is up to the transform implementation to silently do the string=>type
      conversions. That's why you may see e.g. int(arg) in some of the methods.
  (5) Unless it is documented to do so, a transform function must not raise any
      exceptions related to the resource r. The `undefined' arg is used to
      handle all unusual conditions, including ones that would raise exceptions.
      Exceptions for arguments explicitly under the caller's control are OK.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals


def _TransformState(data, undefined=''):
  """Returns textual information about functions state.

  Args:
    data: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    str containing information about the functions state.
  """
  if not isinstance(data, dict):
    return undefined
  if 'status' in data:
    return data['status']
  if 'state' in data:
    return data['state']
  return undefined


def TransformTrigger(data, undefined=''):
  """Returns textual information about functions trigger.

  Args:
    data: JSON-serializable 1st and 2nd gen Functions objects.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    str containing information about functions trigger.
  """
  generation = _TransformGeneration(data)
  if generation == '1st gen':
    if 'httpsTrigger' in data:
      return 'HTTP Trigger'
    if 'gcsTrigger' in data:
      return 'bucket: ' + data['gcsTrigger']
    if 'pubsubTrigger' in data:
      return 'topic: ' + data['pubsubTrigger'].split('/')[-1]
    if 'eventTrigger' in data:
      return 'Event Trigger'
    return undefined

  elif generation == '2nd gen':
    if 'eventTrigger' in data:
      event_trigger = data['eventTrigger']
      if 'pubsubTopic' in event_trigger:
        return 'topic: ' + event_trigger['pubsubTopic'].split('/')[-1]
      return 'Event Trigger'

    # v2 functions can always be http triggered as backed by a cloud run
    # service, if no trigger is found display 'HTTP trigger'
    return 'HTTP Trigger'

  return undefined


def _TransformGeneration(data, undefined='-'):
  """Returns Cloud Functions product version.

  Args:
    data: JSON-serializable 1st and 2nd gen Functions objects.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    str containing inferred product version.
  """

  # data.get returns None if entry doesn't exist
  entry_point = data.get('entryPoint')
  build_id = data.get('buildId')
  runtime = data.get('runtime')
  if any([entry_point, build_id, runtime]):
    return '1st gen'

  build_config = data.get('buildConfig')
  service_config = data.get('serviceConfig')

  if any([build_config, service_config]):
    return '2nd gen'

  return undefined


_TRANSFORMS = {
    'trigger': TransformTrigger,
    'state': _TransformState,
    'generation': _TransformGeneration,
}


def GetTransforms():
  """Returns the functions specific resource transform symbol table."""
  return _TRANSFORMS
