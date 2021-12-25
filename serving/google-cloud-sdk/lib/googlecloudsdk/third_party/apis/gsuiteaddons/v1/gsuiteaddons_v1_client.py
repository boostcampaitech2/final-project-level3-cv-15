"""Generated client library for gsuiteaddons version v1."""
# NOTE: This file is autogenerated and should not be edited by hand.

from __future__ import absolute_import

from apitools.base.py import base_api
from googlecloudsdk.third_party.apis.gsuiteaddons.v1 import gsuiteaddons_v1_messages as messages


class GsuiteaddonsV1(base_api.BaseApiClient):
  """Generated client library for service gsuiteaddons version v1."""

  MESSAGES_MODULE = messages
  BASE_URL = 'https://gsuiteaddons.googleapis.com/'
  MTLS_BASE_URL = 'https://gsuiteaddons.mtls.googleapis.com/'

  _PACKAGE = 'gsuiteaddons'
  _SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
  _VERSION = 'v1'
  _CLIENT_ID = '1042881264118.apps.googleusercontent.com'
  _CLIENT_SECRET = 'x_Tw5K8nnjoRAqULM9PFAC2b'
  _USER_AGENT = 'google-cloud-sdk'
  _CLIENT_CLASS_NAME = 'GsuiteaddonsV1'
  _URL_VERSION = 'v1'
  _API_KEY = None

  def __init__(self, url='', credentials=None,
               get_credentials=True, http=None, model=None,
               log_request=False, log_response=False,
               credentials_args=None, default_global_params=None,
               additional_http_headers=None, response_encoding=None):
    """Create a new gsuiteaddons handle."""
    url = url or self.BASE_URL
    super(GsuiteaddonsV1, self).__init__(
        url, credentials=credentials,
        get_credentials=get_credentials, http=http, model=model,
        log_request=log_request, log_response=log_response,
        credentials_args=credentials_args,
        default_global_params=default_global_params,
        additional_http_headers=additional_http_headers,
        response_encoding=response_encoding)
    self.projects_deployments = self.ProjectsDeploymentsService(self)
    self.projects = self.ProjectsService(self)

  class ProjectsDeploymentsService(base_api.BaseApiService):
    """Service class for the projects_deployments resource."""

    _NAME = 'projects_deployments'

    def __init__(self, client):
      super(GsuiteaddonsV1.ProjectsDeploymentsService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      r"""Creates a deployment with the specified name and configuration.

      Args:
        request: (GsuiteaddonsProjectsDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1Deployment) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/deployments',
        http_method='POST',
        method_id='gsuiteaddons.projects.deployments.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['deploymentId'],
        relative_path='v1/{+parent}/deployments',
        request_field='googleCloudGsuiteaddonsV1Deployment',
        request_type_name='GsuiteaddonsProjectsDeploymentsCreateRequest',
        response_type_name='GoogleCloudGsuiteaddonsV1Deployment',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes the deployment with the given name.

      Args:
        request: (GsuiteaddonsProjectsDeploymentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/deployments/{deploymentsId}',
        http_method='DELETE',
        method_id='gsuiteaddons.projects.deployments.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['etag'],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='GsuiteaddonsProjectsDeploymentsDeleteRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets the deployment with the specified name.

      Args:
        request: (GsuiteaddonsProjectsDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1Deployment) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/deployments/{deploymentsId}',
        http_method='GET',
        method_id='gsuiteaddons.projects.deployments.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='GsuiteaddonsProjectsDeploymentsGetRequest',
        response_type_name='GoogleCloudGsuiteaddonsV1Deployment',
        supports_download=False,
    )

    def GetInstallStatus(self, request, global_params=None):
      r"""Gets the install status of a test deployment.

      Args:
        request: (GsuiteaddonsProjectsDeploymentsGetInstallStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1InstallStatus) The response message.
      """
      config = self.GetMethodConfig('GetInstallStatus')
      return self._RunMethod(
          config, request, global_params=global_params)

    GetInstallStatus.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/deployments/{deploymentsId}/installStatus',
        http_method='GET',
        method_id='gsuiteaddons.projects.deployments.getInstallStatus',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='GsuiteaddonsProjectsDeploymentsGetInstallStatusRequest',
        response_type_name='GoogleCloudGsuiteaddonsV1InstallStatus',
        supports_download=False,
    )

    def Install(self, request, global_params=None):
      r"""Installs a deployment to your account for testing. For more information, see [Test your add-on](https://developers.google.com/workspace/add-ons/guides/alternate-runtimes#test_your_add-on).

      Args:
        request: (GsuiteaddonsProjectsDeploymentsInstallRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('Install')
      return self._RunMethod(
          config, request, global_params=global_params)

    Install.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/deployments/{deploymentsId}:install',
        http_method='POST',
        method_id='gsuiteaddons.projects.deployments.install',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:install',
        request_field='googleCloudGsuiteaddonsV1InstallDeploymentRequest',
        request_type_name='GsuiteaddonsProjectsDeploymentsInstallRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists all deployments in a particular project.

      Args:
        request: (GsuiteaddonsProjectsDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1ListDeploymentsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/deployments',
        http_method='GET',
        method_id='gsuiteaddons.projects.deployments.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1/{+parent}/deployments',
        request_field='',
        request_type_name='GsuiteaddonsProjectsDeploymentsListRequest',
        response_type_name='GoogleCloudGsuiteaddonsV1ListDeploymentsResponse',
        supports_download=False,
    )

    def ReplaceDeployment(self, request, global_params=None):
      r"""Creates or replaces a deployment with the specified name.

      Args:
        request: (GsuiteaddonsProjectsDeploymentsReplaceDeploymentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1Deployment) The response message.
      """
      config = self.GetMethodConfig('ReplaceDeployment')
      return self._RunMethod(
          config, request, global_params=global_params)

    ReplaceDeployment.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/deployments/{deploymentsId}',
        http_method='PUT',
        method_id='gsuiteaddons.projects.deployments.replaceDeployment',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='googleCloudGsuiteaddonsV1Deployment',
        request_type_name='GsuiteaddonsProjectsDeploymentsReplaceDeploymentRequest',
        response_type_name='GoogleCloudGsuiteaddonsV1Deployment',
        supports_download=False,
    )

    def Uninstall(self, request, global_params=None):
      r"""Uninstalls a test deployment from the user's account. For more information, see [Test your add-on](https://developers.google.com/workspace/add-ons/guides/alternate-runtimes#test_your_add-on).

      Args:
        request: (GsuiteaddonsProjectsDeploymentsUninstallRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('Uninstall')
      return self._RunMethod(
          config, request, global_params=global_params)

    Uninstall.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/deployments/{deploymentsId}:uninstall',
        http_method='POST',
        method_id='gsuiteaddons.projects.deployments.uninstall',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}:uninstall',
        request_field='googleCloudGsuiteaddonsV1UninstallDeploymentRequest',
        request_type_name='GsuiteaddonsProjectsDeploymentsUninstallRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

  class ProjectsService(base_api.BaseApiService):
    """Service class for the projects resource."""

    _NAME = 'projects'

    def __init__(self, client):
      super(GsuiteaddonsV1.ProjectsService, self).__init__(client)
      self._upload_configs = {
          }

    def GetAuthorization(self, request, global_params=None):
      r"""Gets the authorization information for deployments in a given project.

      Args:
        request: (GsuiteaddonsProjectsGetAuthorizationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1Authorization) The response message.
      """
      config = self.GetMethodConfig('GetAuthorization')
      return self._RunMethod(
          config, request, global_params=global_params)

    GetAuthorization.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1/projects/{projectsId}/authorization',
        http_method='GET',
        method_id='gsuiteaddons.projects.getAuthorization',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1/{+name}',
        request_field='',
        request_type_name='GsuiteaddonsProjectsGetAuthorizationRequest',
        response_type_name='GoogleCloudGsuiteaddonsV1Authorization',
        supports_download=False,
    )
