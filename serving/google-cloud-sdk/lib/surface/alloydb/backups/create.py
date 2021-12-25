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
"""Creates a new AlloyDB backup."""


from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.api_lib.alloydb import backup_operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources


def _ParseBackupType(alloydb_messages, backup_type):
  if backup_type:
    return alloydb_messages.Backup.TypeValueValuesEnum.lookup_by_name(
        backup_type.upper())
  return None


@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Create(base.CreateCommand):
  """Creates a new AlloyDB backup within a given project."""

  @staticmethod
  def Args(parser):
    """Specifies additional command flags.

    Args:
      parser: argparse.Parser: Parser object for command line inputs.
    """
    base.ASYNC_FLAG.AddToParser(parser)
    parser.add_argument(
        '--region',
        required=True,
        type=str,
        help=('The region of the cluster to backup. Note: both the cluster '
              'and the backup have to be in the same region.'))
    flags.AddBackup(parser)
    flags.AddCluster(parser, False)

  def Run(self, args):
    """Constructs and sends request.

    Args:
      args: argparse.Namespace, An object that contains the values for the
          arguments specified in the .Args() method.

    Returns:
      ProcessHttpResponse of the request made.
    """
    client = api_util.AlloyDBClient(api_util.API_VERSION_DEFAULT)
    alloydb_client = client.alloydb_client
    alloydb_messages = client.alloydb_messages
    location_ref = client.resource_parser.Create(
        'alloydb.projects.locations',
        projectsId=properties.VALUES.core.project.GetOrFail,
        locationsId=args.region)
    cluster_ref = client.resource_parser.Create(
        'alloydb.projects.locations.clusters',
        projectsId=properties.VALUES.core.project.GetOrFail,
        locationsId=args.region,
        clustersId=args.cluster)
    backup_ref = client.resource_parser.Create(
        'alloydb.projects.locations.backups',
        projectsId=properties.VALUES.core.project.GetOrFail,
        locationsId=args.region,
        backupsId=args.backup)

    backup_resource = alloydb_messages.Backup()
    backup_resource.name = backup_ref.RelativeName()
    backup_resource.type = _ParseBackupType(alloydb_messages, 'ON_DEMAND')
    backup_resource.clusterName = cluster_ref.RelativeName()

    req = alloydb_messages.AlloydbProjectsLocationsBackupsCreateRequest(
        backup=backup_resource,
        backupId=args.backup,
        parent=location_ref.RelativeName())
    op = alloydb_client.projects_locations_backups.Create(req)
    op_ref = resources.REGISTRY.ParseRelativeName(
        op.name, collection='alloydb.projects.locations.operations')
    log.status.Print('Operation ID: {}'.format(op_ref.Name()))
    if not args.async_:
      backup_operations.Await(op_ref, 'Creating backup')
    return op
