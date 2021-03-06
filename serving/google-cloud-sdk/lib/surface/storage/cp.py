# -*- coding: utf-8 -*- #
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Implementation of Unix-like cp command for cloud storage providers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.api_lib.storage import user_request_args_factory
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import name_expansion
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_iterator
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files


class Cp(base.Command):
  """Upload, download, and copy Cloud Storage objects."""

  detailed_help = {
      'DESCRIPTION':
          """
      Copy data between your local file system and the cloud, within the cloud,
      and between cloud storage providers.
      """,
      'EXAMPLES':
          """

      The following command uploads all text files from the local directory to a
      bucket:

        $ {command} *.txt gs://my-bucket

      The following command downloads all text files from a bucket to your
      current directory:

        $ {command} gs://my-bucket/*.txt .

      The following command transfers all text files from a bucket to a
      different cloud storage provider:

        $ {command} gs://my-bucket/*.txt s3://my-bucket

      Use the `--recursive` option to copy an entire directory tree. The
      following command uploads the directory tree ``dir'':

        $ {command} --recursive dir gs://my-bucket
      """,
  }

  @staticmethod
  def Args(parser):
    parser.add_argument('source', nargs='+', help='The source path(s) to copy.')
    parser.add_argument('destination', help='The destination path.')
    parser.add_argument(
        '-R',
        '-r',
        '--recursive',
        action='store_true',
        help='Recursively copy the contents of any directories that match the'
        ' source path expression.')
    parser.add_argument(
        '--do-not-decompress',
        action='store_true',
        help='Do not automatically decompress downloaded gzip files.')
    parser.add_argument(
        '--ignore-symlinks',
        action='store_true',
        help='Ignore file symlinks instead of copying what they point to.'
        ' Symlinks pointing to directories will always be ignored.')
    flags.add_precondition_flags(parser)
    flags.add_object_metadata_flags(parser)
    flags.add_encryption_flags(parser)

  def Run(self, args):
    encryption_util.initialize_key_store(args)
    source_expansion_iterator = name_expansion.NameExpansionIterator(
        args.source,
        recursion_requested=args.recursive,
        ignore_symlinks=args.ignore_symlinks)
    task_status_queue = task_graph_executor.multiprocessing_context.Queue()

    raw_destination_url = storage_url.storage_url_from_string(args.destination)
    if (isinstance(raw_destination_url, storage_url.FileUrl) and
        raw_destination_url.is_pipe):
      log.warning('Downloading to a pipe.'
                  ' This command may stall until the pipe is read.')
      shared_stream = files.BinaryFileWriter(args.destination)
      parallelizable = False
    else:
      shared_stream = None
      parallelizable = True

    user_request_args = (
        user_request_args_factory.get_user_request_args_from_command_args(
            args, metadata_type=user_request_args_factory.MetadataType.OBJECT))
    task_iterator = copy_task_iterator.CopyTaskIterator(
        source_expansion_iterator,
        args.destination,
        custom_md5_digest=args.content_md5,
        do_not_decompress=args.do_not_decompress,
        shared_stream=shared_stream,
        task_status_queue=task_status_queue,
        user_request_args=user_request_args,
    )
    self.exit_code = task_executor.execute_tasks(
        task_iterator,
        parallelizable=parallelizable,
        task_status_queue=task_status_queue,
        progress_type=task_status.ProgressType.FILES_AND_BYTES,
    )

    if shared_stream:
      shared_stream.close()
