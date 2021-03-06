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
"""services vpc-peerings delete command."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.services import peering
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties

OP_BASE_CMD = 'gcloud services vpc-peerings operations '
OP_WAIT_CMD = OP_BASE_CMD + 'wait {0}'


@base.ReleaseTracks(
    base.ReleaseTrack.ALPHA,
    base.ReleaseTrack.BETA,
    base.ReleaseTrack.GA,
)
class Delete(base.SilentCommand):
  """Delete a private service connection to a service for a project network."""

  detailed_help = {
      'DESCRIPTION':
          """\
          This command deletes a private service connection to a service via a
          VPC network.
          """,
      'EXAMPLES':
          """\
          To delete an existing connection for a network called `my-network` on
          the current project to a service called `your-service` run:

            $ {command} --network=my-network --service=your-service

          To run the same command asynchronously (non-blocking), run:

            $ {command} --network=my-network --service=your-service --async
          """,
  }

  @staticmethod
  def Args(parser):
    """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
    parser.add_argument(
        '--network',
        metavar='NETWORK',
        required=True,
        help='The network in the current project which is peered with the service'
    )
    parser.add_argument(
        '--service',
        metavar='SERVICE',
        default='servicenetworking.googleapis.com',
        help='The service to connect to')
    base.ASYNC_FLAG.AddToParser(parser)

  def Run(self, args):
    """Run 'services vpc-peerings delete'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      Nothing.
    """
    project = properties.VALUES.core.project.Get(required=True)
    project_number = _GetProjectNumber(project)

    op = peering.DeleteConnection(project_number, args.service, args.network)
    if args.async_:
      cmd = OP_WAIT_CMD.format(op.name)
      log.status.Print('Asynchronous operation is in progress... '
                       'Use the following command to wait for its '
                       'completion:\n {0}'.format(cmd))
      return
    op = services_util.WaitOperation(op.name, peering.GetOperation)
    services_util.PrintOperation(op)


def _GetProjectNumber(project_id):
  return projects_api.Get(projects_util.ParseProject(project_id)).projectNumber
