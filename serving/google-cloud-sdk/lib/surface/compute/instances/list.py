# -*- coding: utf-8 -*- #
# Copyright 2014 Google LLC. All Rights Reserved.
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
"""Command for listing instances."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.instances import flags

RESOURCE_TYPE = 'instances'

DETAILED_HELP = {
    'brief':
        'List Compute Engine ' + RESOURCE_TYPE,
    'DESCRIPTION':
        """\
          *{{command}}* displays all Compute Engine {0} in a project.
        """.format(RESOURCE_TYPE)
}

EXAMPLE_FORMAT = """\
          To list all {0} in a project in table form, run:

            $ {{command}}

      To list the URIs of all {0} in a project, run:

            $ {{command}} --uri

      To list the IPv6 info of all {0} in a project, run:

            $ {{command}} --format="{1}"
    """


class List(base.ListCommand):
  """List Compute Engine virtual machine instances."""

  @staticmethod
  def Args(parser):
    parser.display_info.AddFormat(flags.DEFAULT_LIST_FORMAT)
    parser.display_info.AddUriFunc(utils.MakeGetUriFunc())
    lister.AddZonalListerArgs(parser)
    parser.display_info.AddCacheUpdater(completers.InstancesCompleter)

  def Run(self, args):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    client = holder.client

    request_data = lister.ParseMultiScopeFlags(args, holder.resources)

    list_implementation = lister.MultiScopeLister(
        client=client,
        zonal_service=client.apitools_client.instances,
        aggregation_service=client.apitools_client.instances)

    return lister.Invoke(request_data, list_implementation)


List.detailed_help = DETAILED_HELP.copy()
List.detailed_help['EXAMPLES'] = EXAMPLE_FORMAT.format(
    RESOURCE_TYPE, flags.IPV6_INFO_LIST_FORMAT)
