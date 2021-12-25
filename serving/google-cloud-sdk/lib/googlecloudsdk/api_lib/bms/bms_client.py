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
"""Cloud Bare Metal Solution client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import io
import re

from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer

import six

_DEFAULT_API_VERSION = 'v2'
_GLOBAL_REGION = 'global'
_REGIONAL_IAM_REGEX = re.compile(
    "PERMISSION_DENIED: Permission (.+) denied on 'projects/(.+?)/.*")


def _ParseError(error):
  """Returns a best-effort error message created from an API client error."""
  if isinstance(error, apitools_exceptions.HttpError):
    parsed_error = apilib_exceptions.HttpException(error,
                                                   error_format='{message}')
    error_message = parsed_error.message
  else:
    error_message = six.text_type(error)
  return error_message


def _CollapseRegionalIAMErrors(errors):
  """If all errors are PERMISSION_DENIEDs, use a single global error instead."""
  # TODO(b/198857865): Remove this hack once the `global` region fix is in
  if errors:
    matches = [_REGIONAL_IAM_REGEX.match(e) for e in errors]
    if (all(match is not None for match in matches)
        and len(set(match.group(1) for match in matches)) == 1):
      errors = ['PERMISSION_DENIED: Permission %s denied on projects/%s' %
                (matches[0].group(1), matches[0].group(2))]
  return errors


class BmsClient(object):
  """Cloud Bare Metal Solution client."""

  def __init__(self, api_version=_DEFAULT_API_VERSION):
    self._client = apis.GetClientInstance('baremetalsolution', api_version)
    self._messages = apis.GetMessagesModule('baremetalsolution', api_version)
    self.service = self._client.projects_locations_instances
    self.volumes_service = self._client.projects_locations_volumes
    self.snapshot_schedule_policies_service = self._client.projects_locations_snapshotSchedulePolicies
    self.snapshots_service = self._client.projects_locations_volumes_snapshots
    self.networks_service = self._client.projects_locations_networks
    self.locations_service = self._client.projects_locations
    self.luns_service = self._client.projects_locations_volumes_luns

  @property
  def client(self):
    return self._client

  @property
  def messages(self):
    return self._messages

  def Get(self, resource):
    request = self.messages.BaremetalsolutionProjectsLocationsInstancesGetRequest(
        name=resource.RelativeName())
    return self.service.Get(request)

  def AggregateYieldFromList(self,
                             service,
                             project_resource,
                             request_class,
                             resource,
                             global_params=None,
                             limit=None,
                             method='List',
                             predicate=None,
                             skip_global_region=True,
                             allow_partial_server_failure=True):
    """Make a series of List requests, across locations in a project.

    Args:
      service: apitools_base.BaseApiService, A service with a .List() method.
      project_resource: str, The resource name of the project.
      request_class: class, The class type of the List RPC request.
      resource: string, The name (in plural) of the resource type.
      global_params: protorpc.messages.Message, The global query parameters to
        provide when calling the given method.
      limit: int, The maximum number of records to yield. None if all available
        records should be yielded.
      method: str, The name of the method used to fetch resources.
      predicate: lambda, A function that returns true for items to be yielded.
      skip_global_region: bool, True if global region must be filtered out while
      iterating over regions
      allow_partial_server_failure: bool, if True don't fail and only print a
        warning if some requests fail as long as at least one succeeds. If
        False, fail the complete command if at least one request fails.

    Yields:
      protorpc.message.Message, The resources listed by the service.

    """
    response_count = 0
    errors = []
    for location in self.ListLocations(project_resource):
      # TODO (b/198857865): Global region will be used when it is ready.
      location_name = location.name.split('/')[-1]
      if skip_global_region and location_name == _GLOBAL_REGION:
        continue
      request = request_class(parent=location.name)
      try:
        response = getattr(service, method)(
            request, global_params=global_params)
        response_count += 1
      except Exception as e:  # pylint: disable=broad-except
        errors.append(_ParseError(e))
        continue
      items = getattr(response, resource)
      if predicate:
        items = list(filter(predicate, items))
      for item in items:
        yield item
        if limit is None:
          continue
        limit -= 1
        if not limit:
          break

    if errors:
      # If the command allows partial server errors, instead of raising an
      # exception to show something went wrong, we show a warning message that
      # contains the error messages instead.
      buf = io.StringIO()
      fmt = ('list[title="Some requests did not succeed.",'
             'always-display-title]')
      if allow_partial_server_failure and response_count > 0:
        resource_printer.Print(sorted(set(errors)), fmt, out=buf)
        log.warning(buf.getvalue())
      else:
        # If all requests failed, clean them up if they're duplicated IAM errors
        collapsed_errors = _CollapseRegionalIAMErrors(errors)
        resource_printer.Print(sorted(set(collapsed_errors)), fmt, out=buf)
        raise exceptions.Error(buf.getvalue())

  def ListLocations(self,
                    project_resource,
                    limit=None,
                    page_size=None):
    request = self.messages.BaremetalsolutionProjectsLocationsListRequest(
        name='projects/' + project_resource)
    return list_pager.YieldFromList(
        self.locations_service,
        request,
        limit=limit,
        batch_size_attribute='pageSize',
        batch_size=page_size,
        field='locations')

  def List(self,
           location_resource,
           limit=None,
           page_size=None):
    location = location_resource.RelativeName()
    request = self.messages.BaremetalsolutionProjectsLocationsInstancesListRequest(
        parent=location)
    return list_pager.YieldFromList(
        self.service,
        request,
        limit=limit,
        batch_size_attribute='pageSize',
        batch_size=page_size,
        field='instances')

  def AggregateListInstances(self, project_resource, limit=None):
    return self.AggregateYieldFromList(
        self.service,
        project_resource,
        self.messages.BaremetalsolutionProjectsLocationsInstancesListRequest,
        'instances',
        limit=limit)

  def ListSnapshotSchedulePolicies(self,
                                   project_resource,
                                   limit=None,
                                   page_size=None):
    parent = 'projects/%s/locations/global' % project_resource
    request = (
        self.messages
        .BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesListRequest(
            parent=parent))
    return list_pager.YieldFromList(
        self.snapshot_schedule_policies_service,
        request,
        limit=limit,
        batch_size_attribute='pageSize',
        batch_size=page_size,
        field='snapshotSchedulePolicies')

  def GetSnapshotSchedulePolicy(self, resource):
    request = self.messages.BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesGetRequest(
        name=resource.RelativeName())
    return self.snapshot_schedule_policies_service.Get(request)

  def CreateSnapshotSchedulePolicy(self,
                                   policy_resource,
                                   description,
                                   schedules):
    """Sends request to create a new Snapshot Schedule Policy."""
    policy_id = policy_resource.Name()
    parent = policy_resource.Parent().RelativeName()
    schedule_msgs = self._ParseSnapshotSchedules(schedules)
    policy_msg = self.messages.SnapshotSchedulePolicy(
        description=description, schedules=schedule_msgs)
    request = self.messages.BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesCreateRequest(
        parent=parent,
        snapshotSchedulePolicyId=policy_id,
        snapshotSchedulePolicy=policy_msg)
    return self.snapshot_schedule_policies_service.Create(request)

  def UpdateSnapshotSchedulePolicy(self,
                                   policy_resource,
                                   description,
                                   schedules):
    """Sends request to update an existing SnapshotSchedulePolicy."""
    updated_fields = []
    if description:
      updated_fields.append('description')

    schedule_msgs = self._ParseSnapshotSchedules(schedules)
    if schedule_msgs:
      updated_fields.append('schedules')

    update_mask = ','.join(updated_fields)
    policy_msg = self.messages.SnapshotSchedulePolicy(
        description=description, schedules=schedule_msgs)
    request = self.messages.BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesPatchRequest(
        name=policy_resource.RelativeName(),
        snapshotSchedulePolicy=policy_msg,
        updateMask=update_mask)
    return self.snapshot_schedule_policies_service.Patch(request)

  def _ParseSnapshotSchedules(self, schedules):
    """Parses schedule ArgDict dicts into a list of Schedule messages."""
    schedule_msgs = []
    if schedules:
      for schedule_arg in schedules:
        schedule_msgs.append(self.messages.Schedule(
            crontabSpec=schedule_arg['crontab_spec'],
            retentionCount=schedule_arg['retention_count'],
            prefix=schedule_arg['prefix']))
    return schedule_msgs

  def DeleteSnapshotSchedulePolicy(self, resource):
    request = self.messages.BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesDeleteRequest(
        name=resource.RelativeName())
    return self.snapshot_schedule_policies_service.Delete(request)

  def ListVolumes(self,
                  location_resource,
                  limit=None,
                  page_size=None):
    location = location_resource.RelativeName()
    request = self.messages.BaremetalsolutionProjectsLocationsVolumesListRequest(
        parent=location)
    return list_pager.YieldFromList(
        self.volumes_service,
        request,
        limit=limit,
        batch_size_attribute='pageSize',
        batch_size=page_size,
        field='volumes')

  def UpdateVolume(self,
                   volume_resource,
                   snapshot_schedule_policy_resource,
                   remove_snapshot_schedule_policy,
                   snapshot_auto_delete):
    """Update an existing volume resource."""
    updated_fields = []
    policy_name = None
    if snapshot_schedule_policy_resource:
      updated_fields.append('snapshotSchedulePolicy')
      policy_name = snapshot_schedule_policy_resource.RelativeName()
    elif remove_snapshot_schedule_policy:
      updated_fields.append('snapshotSchedulePolicy')

    if snapshot_auto_delete:
      updated_fields.append('snapshotAutoDeleteBehavior')

    volume_msg = self.messages.Volume(
        name=volume_resource.RelativeName(),
        snapshotAutoDeleteBehavior=snapshot_auto_delete,
        snapshotSchedulePolicy=policy_name)

    request = self.messages.BaremetalsolutionProjectsLocationsVolumesPatchRequest(
        name=volume_resource.RelativeName(),
        volume=volume_msg,
        updateMask=','.join(updated_fields))

    return self.volumes_service.Patch(request)

  def GetVolume(self, resource):
    request = self.messages.BaremetalsolutionProjectsLocationsVolumesGetRequest(
        name=resource.RelativeName())
    return self.volumes_service.Get(request)

  def AggregateListVolumes(self, project_resource, limit=None):
    return self.AggregateYieldFromList(
        self.volumes_service,
        project_resource,
        self.messages.BaremetalsolutionProjectsLocationsVolumesListRequest,
        'volumes',
        limit=limit)

  def ListNetworks(self,
                   location_resource,
                   limit=None,
                   page_size=None):
    location = location_resource.RelativeName()
    request = (
        self.messages
        .BaremetalsolutionProjectsLocationsNetworksListRequest(
            parent=location))
    return list_pager.YieldFromList(
        self.networks_service,
        request,
        limit=limit,
        batch_size_attribute='pageSize',
        batch_size=page_size,
        field='networks')

  def AggregateListNetworks(self, project_resource, limit=None):
    return self.AggregateYieldFromList(
        self.networks_service,
        project_resource,
        self.messages.BaremetalsolutionProjectsLocationsNetworksListRequest,
        'networks',
        limit=limit)

  def GetNetwork(self, resource):
    request = self.messages.BaremetalsolutionProjectsLocationsNetworksGetRequest(
        name=resource.RelativeName())
    return self.networks_service.Get(request)

  def IsClientNetwork(self, network):
    return network.type == self.messages.Network.TypeValueValuesEnum.CLIENT

  def IsPrivateNetwork(self, network):
    return network.type == self.messages.Network.TypeValueValuesEnum.PRIVATE

  def ListLUNsForVolume(self, volume_resource, limit=None,
                        page_size=None):
    parent = volume_resource.RelativeName()
    request = (self.messages
               .BaremetalsolutionProjectsLocationsVolumesLunsListRequest(
                   parent=parent))
    return list_pager.YieldFromList(
        self.luns_service,
        request,
        limit=limit,
        batch_size_attribute='pageSize',
        batch_size=page_size,
        field='luns')

  def GetLUN(self, resource):
    request = (self.messages
               .BaremetalsolutionProjectsLocationsVolumesLunsGetRequest(
                   name=resource.RelativeName()))
    return self.luns_service.Get(request)

  def ListSnapshotsForVolume(self,
                             volume_resource,
                             limit=None,
                             page_size=None):
    parent = volume_resource.RelativeName()
    request = (self.messages
               .BaremetalsolutionProjectsLocationsVolumesSnapshotsListRequest(
                   parent=parent))
    return list_pager.YieldFromList(
        self.snapshots_service,
        request,
        limit=limit,
        batch_size_attribute='pageSize',
        batch_size=page_size,
        field='volumeSnapshots')

  def GetVolumeSnapshot(self, resource):
    request = (self.messages
               .BaremetalsolutionProjectsLocationsVolumesSnapshotsGetRequest(
                   name=resource.RelativeName()))
    return self.snapshots_service.Get(request)

  def CreateVolumeSnapshot(self, resource, description):
    request = (self.messages
               .BaremetalsolutionProjectsLocationsVolumesSnapshotsCreateRequest(
                   parent=resource.RelativeName(),
                   volumeSnapshot=self.messages.VolumeSnapshot(
                       description=description)))
    return self.snapshots_service.Create(request)

  def DeleteVolumeSnapshot(self, resource):
    request = (self.messages
               .BaremetalsolutionProjectsLocationsVolumesSnapshotsDeleteRequest(
                   name=resource.RelativeName()))
    return self.snapshots_service.Delete(request)

  def RestoreVolumeSnapshot(self, volume_resource, snapshot_name):
    snapshot_resource_name = '%s/snapshots/%s' % (
        volume_resource.RelativeName(), snapshot_name)
    request = (
        self.messages
        .BaremetalsolutionProjectsLocationsVolumesSnapshotsRestoreVolumeSnapshotRequest(  # pylint: disable=line-too-long
            volumeSnapshot=snapshot_resource_name))
    return self.snapshots_service.RestoreVolumeSnapshot(request)
