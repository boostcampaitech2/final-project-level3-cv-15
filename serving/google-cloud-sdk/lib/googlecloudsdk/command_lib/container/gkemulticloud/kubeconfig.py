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
"""Utilities for generating kubeconfig entries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import base64


from googlecloudsdk.api_lib.container import kubeconfig as kubeconfig_util
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.container.gkemulticloud import errors
from googlecloudsdk.command_lib.container.hub import connect_gateway_util
from googlecloudsdk.command_lib.container.hub import gwkubeconfig_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import semver


COMMAND_DESCRIPTION = """
Fetch credentials for a running Anthos cluster on {kind}.

This command updates a kubeconfig file with appropriate credentials and
endpoint information to point kubectl at a specific cluster on {kind}.

By default, credentials are written to ``HOME/.kube/config''.
You can provide an alternate path by setting the ``KUBECONFIG'' environment
variable. If ``KUBECONFIG'' contains multiple paths, the first one is used.

This command enables switching to a specific cluster, when working
with multiple clusters. It can also be used to access a previously created
cluster from a new workstation.

By default, the command will configure kubectl to automatically refresh its
credentials using the same identity as the gcloud command-line tool.
If you are running kubectl as part of an application, it is recommended to use
[application default credentials](https://cloud.google.com/docs/authentication/production).
To configure a kubeconfig file to use application default credentials, set
the ``container/use_application_default_credentials''
[Cloud SDK property](https://cloud.google.com/sdk/docs/properties) to ``true''
before running the command.

See [](https://cloud.google.com/kubernetes-engine/docs/kubectl) for
kubectl documentation.
"""

COMMAND_EXAMPLE = """
To get credentials of a cluster named ``my-cluster'' managed in location ``us-west1'',
run:

$ {command} my-cluster --location=us-west1
"""


def GenerateContext(kind, project_id, location, cluster_id):
  """Generates a kubeconfig context for an Anthos Multi-Cloud cluster.

  Args:
    kind: str, kind of the cluster e.g. aws, azure.
    project_id: str, project ID accociated with the cluster.
    location: str, Google location of the cluster.
    cluster_id: str, ID of the cluster.

  Returns:
    The context for the kubeconfig entry.
  """
  template = 'gke_{kind}_{project_id}_{location}_{cluster_id}'
  return template.format(
      kind=kind,
      project_id=project_id,
      location=location,
      cluster_id=cluster_id)


def GenerateAuthProviderCmdArgs(kind, cluster_id, location):
  """Generates command arguments for kubeconfig's authorization provider.

  Args:
    kind: str, kind of the cluster e.g. aws, azure.
    cluster_id: str, ID of the cluster.
    location: str, Google location of the cluster.

  Returns:
    The command arguments for kubeconfig's authorization provider.
  """
  template = ('container {kind} clusters print-access-token '
              '{cluster_id} --location={location}')
  return template.format(kind=kind, cluster_id=cluster_id, location=location)


def GenerateKubeconfig(cluster, context, cmd_path, cmd_args, private_ep=False):
  """Generates a kubeconfig entry for an Anthos Multi-cloud cluster.

  Args:
    cluster: object, Anthos Multi-cloud cluster.
    context: str, context for the kubeconfig entry.
    cmd_path: str, authentication provider command path.
    cmd_args: str, authentication provider command arguments.
    private_ep: bool, whether to use private VPC for authentication.

  Raises:
      Error: don't have the permission to open kubeconfig file.
  """
  kubeconfig = kubeconfig_util.Kubeconfig.Default()
  # Use the same key for context, cluster, and user.
  kubeconfig.contexts[context] = kubeconfig_util.Context(
      context, context, context)

  # Only default to use Connect Gateway for 1.21+.
  version = _GetSemver(cluster)
  if private_ep or version < semver.SemVer('1.21.0'):
    _CheckPreqs(private_endpoint=True)
    _PrivateVPCKubeconfig(kubeconfig, cluster, context, cmd_path, cmd_args)
  else:
    _CheckPreqs()
    _ConnectGatewayKubeconfig(kubeconfig, cluster, context, cmd_path)

  kubeconfig.SetCurrentContext(context)
  kubeconfig.SaveToFile()
  log.status.Print(
      'A new kubeconfig entry "{}" has been generated and set as the '
      'current context.'.format(context))


def _CheckPreqs(private_endpoint=False):
  """Checks the prerequisites to run get-credentials commands."""
  util.CheckKubectlInstalled()
  if not private_endpoint:
    project_id = properties.VALUES.core.project.GetOrFail()
    connect_gateway_util.CheckGatewayApiEnablement(project_id,
                                                   _GetConnectGatewayEndpoint())


def _ConnectGatewayKubeconfig(kubeconfig, cluster, context, cmd_path):
  """Generates the Connect Gateway kubeconfig entry.

  Args:
    kubeconfig: object, Kubeconfig object.
    cluster: object, Anthos Multi-cloud cluster.
    context: str, context for the kubeconfig entry.
    cmd_path: str, authentication provider command path.

  Raises:
      errors.MissingClusterField: cluster is missing required fields.
  """
  if cluster.fleet is None:
    raise errors.MissingClusterField('fleet')
  if cluster.fleet.membership is None:
    raise errors.MissingClusterField('fleet.membership')
  membership_resource = resources.REGISTRY.ParseRelativeName(
      cluster.fleet.membership,
      collection='gkehub.projects.locations.memberships')
  # Connect Gateway only supports project number.
  # TODO(b/198380839): Use the url with locations once rolled out.
  server = 'https://{}/v1/projects/{}/memberships/{}'.format(
      _GetConnectGatewayEndpoint(),
      project_util.GetProjectNumber(membership_resource.projectsId),
      membership_resource.membershipsId)
  user_kwargs = {'auth_provider': 'gcp', 'auth_provider_cmd_path': cmd_path}
  kubeconfig.users[context] = kubeconfig_util.User(context, **user_kwargs)
  kubeconfig.clusters[context] = gwkubeconfig_util.Cluster(context, server)


def _PrivateVPCKubeconfig(kubeconfig, cluster, context, cmd_path, cmd_args):
  """Generates the kubeconfig entry to connect using private VPC.

  Args:
    kubeconfig: object, Kubeconfig object.
    cluster: object, Anthos Multi-cloud cluster.
    context: str, context for the kubeconfig entry.
    cmd_path: str, authentication provider command path.
    cmd_args: str, authentication provider command arguments.
  """
  user_kwargs = {
      'auth_provider': 'gcp',
      'auth_provider_cmd_path': cmd_path,
      'auth_provider_cmd_args': cmd_args,
      'auth_provider_expiry_key': '{.expirationTime}',
      'auth_provider_token_key': '{.accessToken}'
  }
  kubeconfig.users[context] = kubeconfig_util.User(context, **user_kwargs)

  cluster_kwargs = {}
  if cluster.clusterCaCertificate is None:
    log.warning('Cluster is missing certificate authority data.')
  else:
    cluster_kwargs['ca_data'] = _GetCaData(cluster.clusterCaCertificate)
  kubeconfig.clusters[context] = kubeconfig_util.Cluster(
      context, 'https://{}'.format(cluster.endpoint), **cluster_kwargs)


def ValidateClusterVersion(cluster):
  """Validates the cluster version.

  Args:
    cluster: object, Anthos Multi-cloud cluster.

  Raises:
      UnsupportedClusterVersion: cluster version is not supported.
      MissingClusterField: expected cluster field is missing.
  """
  version = _GetSemver(cluster)
  if version < semver.SemVer('1.20.0'):
    raise errors.UnsupportedClusterVersion(
        'The command get-credentials is supported in cluster version 1.20 '
        'and newer. For older versions, use get-kubeconfig.')


def _GetCaData(pem):
  # Field certificate-authority-data in kubeconfig
  # expects a base64 encoded string of a PEM.
  return base64.b64encode(pem.encode('utf-8')).decode('utf-8')


def _GetSemver(cluster):
  if cluster.controlPlane is None or cluster.controlPlane.version is None:
    raise errors.MissingClusterField('version')
  version = cluster.controlPlane.version
  # The dev version e.g. 1.21-next does not conform to semantic versioning.
  # Replace the -next suffix before parsing semver for version comparison.
  if version.endswith('-next'):
    v = version.replace('-next', '.0', 1)
    return semver.SemVer(v)
  return semver.SemVer(version)


def _GetConnectGatewayEndpoint():
  """Gets the corresponding Connect Gateway endpoint for Multicloud environment.

  http://g3doc/cloud/kubernetes/multicloud/g3doc/oneplatform/team/how-to/hub

  Returns:
    The Connect Gateway endpoint.

  Raises:
    Error: Unknown API override.
  """
  endpoint = properties.VALUES.api_endpoint_overrides.gkemulticloud.Get()
  # Multicloud overrides prod endpoint at run time with the regionalized version
  # so we can't simply check that endpoint is not overriden.
  if endpoint is None or endpoint.endswith(
      'gkemulticloud.googleapis.com/') or endpoint.endswith(
          'preprod-gkemulticloud.sandbox.googleapis.com/'):
    return 'connectgateway.googleapis.com'
  if 'staging-gkemulticloud' in endpoint:
    return 'staging-connectgateway.sandbox.googleapis.com'
  if endpoint.startswith('http://localhost') or endpoint.endswith(
      'gkemulticloud.sandbox.googleapis.com/'):
    return 'autopush-connectgateway.sandbox.googleapis.com'
  raise errors.UnknownApiEndpointOverrideError('gkemulticloud')
