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
"""Task for streaming downloads.

Typically executed in a task iterator:
googlecloudsdk.command_lib.storage.tasks.task_executor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
import threading

from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status


class StreamingDownloadTask(task.Task):
  """Represents a command operation triggering a streaming download."""

  def __init__(self, source_resource, download_stream, user_request_args=None):
    """Initializes task.

    Args:
      source_resource (ObjectResource): Must contain the full path of object to
        download, including bucket. Directories will not be accepted. Does not
        need to contain metadata.
      download_stream (stream): Reusable stream to write download to.
      user_request_args (UserRequestArgs|None): Values for RequestConfig.
    """
    super(StreamingDownloadTask, self).__init__()
    self._source_resource = source_resource
    self._download_stream = download_stream
    self._user_request_args = user_request_args

  def execute(self, task_status_queue=None):
    """Runs download to stream."""
    progress_callback = progress_callbacks.FilesAndBytesProgressCallback(
        status_queue=task_status_queue,
        offset=0,
        length=self._source_resource.size,
        source_url=self._source_resource.storage_url,
        destination_url=self._download_stream.name,
        operation_name=task_status.OperationName.DOWNLOADING,
        process_id=os.getpid(),
        thread_id=threading.get_ident(),
    )

    request_config = request_config_factory.get_request_config(
        self._source_resource.storage_url,
        decryption_key_hash=self._source_resource.decryption_key_hash,
        user_request_args=self._user_request_args,
    )

    provider = self._source_resource.storage_url.scheme
    api_factory.get_api(provider).download_object(
        self._source_resource,
        self._download_stream,
        request_config,
        download_strategy=cloud_api.DownloadStrategy.ONE_SHOT,
        progress_callback=progress_callback)
