# -*- coding: utf-8 -*- #
# Copyright 2016 Google LLC. All Rights Reserved.
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
"""Wrapper for user-visible error exceptions to raise in the CLI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.core import exceptions


class Error(exceptions.Error):
  """Exceptions for Deployment Manager errors."""


class ArgumentError(Error):
  """Command argument error."""


class JobError(Error):
  """Job encountered an error."""


class JobTimeoutError(JobError):
  """Job timed out."""


class OperationError(Error):
  """Operation encountered an error."""


class OperationTimeoutError(OperationError):
  """Operation timed out."""


class ParseError(Error):
  """File parsing error."""


class FileUploadError(Error):
  """File upload error."""


class ObjectReadError(Error):
  """Cloud Storage Object read error."""


class ValidationError(Error):
  """Error while validating YAML against schema."""


class PersonalAuthError(Exception):
  """Error while establishing a personal auth session."""
