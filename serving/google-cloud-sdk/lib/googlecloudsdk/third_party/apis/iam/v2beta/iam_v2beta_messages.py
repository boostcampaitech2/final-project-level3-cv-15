"""Generated message classes for iam version v2beta.

Manages identity and access control for Google Cloud Platform resources,
including the creation of service accounts, which you can use to authenticate
to Google and make API calls.
"""
# NOTE: This file is autogenerated and should not be edited by hand.

from __future__ import absolute_import

from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types


package = 'iam'


class GoogleIamAdminV1AuditData(_messages.Message):
  r"""Audit log information specific to Cloud IAM admin APIs. This message is
  serialized as an `Any` type in the `ServiceData` message of an `AuditLog`
  message.

  Fields:
    permissionDelta: The permission_delta when when creating or updating a
      Role.
  """

  permissionDelta = _messages.MessageField('GoogleIamAdminV1AuditDataPermissionDelta', 1)


class GoogleIamAdminV1AuditDataPermissionDelta(_messages.Message):
  r"""A PermissionDelta message to record the added_permissions and
  removed_permissions inside a role.

  Fields:
    addedPermissions: Added permissions.
    removedPermissions: Removed permissions.
  """

  addedPermissions = _messages.StringField(1, repeated=True)
  removedPermissions = _messages.StringField(2, repeated=True)


class GoogleIamV1BindingDelta(_messages.Message):
  r"""One delta entry for Binding. Each individual change (only one member in
  each entry) to a binding will be a separate entry.

  Enums:
    ActionValueValuesEnum: The action that was performed on a Binding.
      Required

  Fields:
    action: The action that was performed on a Binding. Required
    condition: The condition that is associated with this binding.
    member: A single identity requesting access for a Cloud Platform resource.
      Follows the same format of Binding.members. Required
    role: Role that is assigned to `members`. For example, `roles/viewer`,
      `roles/editor`, or `roles/owner`. Required
  """

  class ActionValueValuesEnum(_messages.Enum):
    r"""The action that was performed on a Binding. Required

    Values:
      ACTION_UNSPECIFIED: Unspecified.
      ADD: Addition of a Binding.
      REMOVE: Removal of a Binding.
    """
    ACTION_UNSPECIFIED = 0
    ADD = 1
    REMOVE = 2

  action = _messages.EnumField('ActionValueValuesEnum', 1)
  condition = _messages.MessageField('GoogleTypeExpr', 2)
  member = _messages.StringField(3)
  role = _messages.StringField(4)


class GoogleIamV1LoggingAuditData(_messages.Message):
  r"""Audit log information specific to Cloud IAM. This message is serialized
  as an `Any` type in the `ServiceData` message of an `AuditLog` message.

  Fields:
    policyDelta: Policy delta between the original policy and the newly set
      policy.
  """

  policyDelta = _messages.MessageField('GoogleIamV1PolicyDelta', 1)


class GoogleIamV1PolicyDelta(_messages.Message):
  r"""The difference delta between two policies.

  Fields:
    bindingDeltas: The delta for Bindings between two policies.
  """

  bindingDeltas = _messages.MessageField('GoogleIamV1BindingDelta', 1, repeated=True)


class GoogleIamV2betaDenyRule(_messages.Message):
  r"""IAM Deny Policy Rule.

  Fields:
    denialCondition: The condition that is associated with this deny rule.
      NOTE: A satisfied condition will explicitly deny access for applicable
      principal, permission, and resource. Different deny rules, including
      their conditions, are examined independently. Only tag based conditions
      are supported.
    deniedPermissions: Specifies the permissions that are explicitly denied by
      this rule. The denied permission can be specified in the following ways:
      * a full permission string in the following canonical formats as
      described in the service specific documentation: --
      `{service_FQDN}/{resource}.{verb}`. For example,
      `iam.googleapis.com/roles.list`.
    deniedPrincipals: A string attribute.
    exceptionPermissions: Specifies the permissions that are excluded from the
      set of denied permissions given by `denied_permissions`. If a permission
      appears in `denied_permissions` _and_ in `excluded_permissions` then it
      will _not_ be denied. The excluded permissions can be specified using
      the same syntax as `denied_permissions`.
    exceptionPrincipals: Specifies the identities requesting access for a
      Cloud Platform resource, which are excluded from the deny rule.
      `exception_principals` can have the following values: * Google and G
      Suite user accounts: -- `principal://goog/subject/{emailId}`: An email
      address that represents a specific Google account. For example,
      `principal://goog/subject/alice@gmail.com`. * Google and G Suite groups:
      -- `principalSet://goog/group/{groupId}`: An identifier that represents
      a Google group. For example,
      `principalSet://goog/group/admins@example.com`. * Service Accounts: -- `
      principal://iam.googleapis.com/projects/-/serviceAccounts/{serviceAccoun
      tId}`: An identifier that represents a service account. For example, `pr
      incipal://iam.googleapis.com/projects/-/serviceAccounts/sa123@iam.gservi
      ceaccount.com`. * G Suite Customers: --
      `principalSet://goog/cloudIdentityCustomerId/{customerId}`: All of the
      principals associated with the specified G Suite Customer ID. For
      example, `principalSet://goog/cloudIdentityCustomerId/C01Abc35`.
  """

  denialCondition = _messages.MessageField('GoogleTypeExpr', 1)
  deniedPermissions = _messages.StringField(2, repeated=True)
  deniedPrincipals = _messages.StringField(3, repeated=True)
  exceptionPermissions = _messages.StringField(4, repeated=True)
  exceptionPrincipals = _messages.StringField(5, repeated=True)


class GoogleIamV2betaListPoliciesResponse(_messages.Message):
  r"""Response message for ListPolicies method.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    policies: The collection of policy metadata that are attached on the
      resource.
  """

  nextPageToken = _messages.StringField(1)
  policies = _messages.MessageField('GoogleIamV2betaPolicy', 2, repeated=True)


class GoogleIamV2betaPolicy(_messages.Message):
  r"""Represents policy data.

  Fields:
    createTime: Output only. The time when the `Policy` was created.
    deleteTime: Output only. The time when the `Policy` was deleted. Empty if
      the policy is not deleted.
    displayName: A user-specified opaque description of the `Policy`. Must be
      less than or equal to 63 characters.
    etag: An opaque tag indicating the current version of the `Policy`, used
      for concurrency control. When the `Policy` is returned from `GetPolicy`
      request, this `etag` indicates the version of the current `Policy` to
      use when executing a read-modify-write loop. When the `Policy` is used
      in a `UpdatePolicy` method, use the `etag` value that was returned from
      a `GetPolicy` request as part of a read-modify-write loop for
      concurrency control. This field is ignored if used in a `CreatePolicy`
      request.
    kind: Output only. The kind of the `Policy`. This is a read only field
      derived from the policy name.
    name: Immutable. The resource name of the `Policy`. Takes the form:
      `policies/{attachment-point}/{kind-plural}/{policy-id}` The attachment
      point is identified by its URL encoded full resource name, which means
      that the forward-slash character, '/', must be written as %2F. For
      example, `policies/cloudresourcemanager.googleapis.com%2Fprojects%2F123/
      denypolicies/a-deny-policy`.
    rules: List of rules that specify the behavior of the `Policy`. The list
      contains a single kind of rules, that matches the kind specified in the
      policy name.
    uid: Immutable. The globally unique ID of the `Policy`. This is a read
      only field assigned on policy creation.
    updateTime: Output only. The time when the `Policy` was last updated.
  """

  createTime = _messages.StringField(1)
  deleteTime = _messages.StringField(2)
  displayName = _messages.StringField(3)
  etag = _messages.StringField(4)
  kind = _messages.StringField(5)
  name = _messages.StringField(6)
  rules = _messages.MessageField('GoogleIamV2betaPolicyRule', 7, repeated=True)
  uid = _messages.StringField(8)
  updateTime = _messages.StringField(9)


class GoogleIamV2betaPolicyOperationMetadata(_messages.Message):
  r"""Metadata for long-running Policy operations.

  Fields:
    createTime: Timestamp when the google.longrunning.Operation was created.
  """

  createTime = _messages.StringField(1)


class GoogleIamV2betaPolicyRule(_messages.Message):
  r"""A single rule in a `Policy`.

  Fields:
    denyRule: Specification of a Deny `Policy`.
    description: A user-specified opaque description of the rule. Must be less
      than or equal to 256 characters.
  """

  denyRule = _messages.MessageField('GoogleIamV2betaDenyRule', 1)
  description = _messages.StringField(2)


class GoogleLongrunningOperation(_messages.Message):
  r"""This resource represents a long-running operation that is the result of
  a network API call.

  Messages:
    MetadataValue: Service-specific metadata associated with the operation. It
      typically contains progress information and common metadata such as
      create time. Some services might not provide such metadata. Any method
      that returns a long-running operation should document the metadata type,
      if any.
    ResponseValue: The normal response of the operation in case of success. If
      the original method returns no data on success, such as `Delete`, the
      response is `google.protobuf.Empty`. If the original method is standard
      `Get`/`Create`/`Update`, the response should be the resource. For other
      methods, the response should have the type `XxxResponse`, where `Xxx` is
      the original method name. For example, if the original method name is
      `TakeSnapshot()`, the inferred response type is `TakeSnapshotResponse`.

  Fields:
    done: If the value is `false`, it means the operation is still in
      progress. If `true`, the operation is completed, and either `error` or
      `response` is available.
    error: The error result of the operation in case of failure or
      cancellation.
    metadata: Service-specific metadata associated with the operation. It
      typically contains progress information and common metadata such as
      create time. Some services might not provide such metadata. Any method
      that returns a long-running operation should document the metadata type,
      if any.
    name: The server-assigned name, which is only unique within the same
      service that originally returns it. If you use the default HTTP mapping,
      the `name` should be a resource name ending with
      `operations/{unique_id}`.
    response: The normal response of the operation in case of success. If the
      original method returns no data on success, such as `Delete`, the
      response is `google.protobuf.Empty`. If the original method is standard
      `Get`/`Create`/`Update`, the response should be the resource. For other
      methods, the response should have the type `XxxResponse`, where `Xxx` is
      the original method name. For example, if the original method name is
      `TakeSnapshot()`, the inferred response type is `TakeSnapshotResponse`.
  """

  @encoding.MapUnrecognizedFields('additionalProperties')
  class MetadataValue(_messages.Message):
    r"""Service-specific metadata associated with the operation. It typically
    contains progress information and common metadata such as create time.
    Some services might not provide such metadata. Any method that returns a
    long-running operation should document the metadata type, if any.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

    class AdditionalProperty(_messages.Message):
      r"""An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """

      key = _messages.StringField(1)
      value = _messages.MessageField('extra_types.JsonValue', 2)

    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

  @encoding.MapUnrecognizedFields('additionalProperties')
  class ResponseValue(_messages.Message):
    r"""The normal response of the operation in case of success. If the
    original method returns no data on success, such as `Delete`, the response
    is `google.protobuf.Empty`. If the original method is standard
    `Get`/`Create`/`Update`, the response should be the resource. For other
    methods, the response should have the type `XxxResponse`, where `Xxx` is
    the original method name. For example, if the original method name is
    `TakeSnapshot()`, the inferred response type is `TakeSnapshotResponse`.

    Messages:
      AdditionalProperty: An additional property for a ResponseValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

    class AdditionalProperty(_messages.Message):
      r"""An additional property for a ResponseValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """

      key = _messages.StringField(1)
      value = _messages.MessageField('extra_types.JsonValue', 2)

    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

  done = _messages.BooleanField(1)
  error = _messages.MessageField('GoogleRpcStatus', 2)
  metadata = _messages.MessageField('MetadataValue', 3)
  name = _messages.StringField(4)
  response = _messages.MessageField('ResponseValue', 5)


class GoogleRpcStatus(_messages.Message):
  r"""The `Status` type defines a logical error model that is suitable for
  different programming environments, including REST APIs and RPC APIs. It is
  used by [gRPC](https://github.com/grpc). Each `Status` message contains
  three pieces of data: error code, error message, and error details. You can
  find out more about this error model and how to work with it in the [API
  Design Guide](https://cloud.google.com/apis/design/errors).

  Messages:
    DetailsValueListEntry: A DetailsValueListEntry object.

  Fields:
    code: The status code, which should be an enum value of google.rpc.Code.
    details: A list of messages that carry the error details. There is a
      common set of message types for APIs to use.
    message: A developer-facing error message, which should be in English. Any
      user-facing error message should be localized and sent in the
      google.rpc.Status.details field, or localized by the client.
  """

  @encoding.MapUnrecognizedFields('additionalProperties')
  class DetailsValueListEntry(_messages.Message):
    r"""A DetailsValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a DetailsValueListEntry
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

    class AdditionalProperty(_messages.Message):
      r"""An additional property for a DetailsValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """

      key = _messages.StringField(1)
      value = _messages.MessageField('extra_types.JsonValue', 2)

    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

  code = _messages.IntegerField(1, variant=_messages.Variant.INT32)
  details = _messages.MessageField('DetailsValueListEntry', 2, repeated=True)
  message = _messages.StringField(3)


class GoogleTypeExpr(_messages.Message):
  r"""Represents a textual expression in the Common Expression Language (CEL)
  syntax. CEL is a C-like expression language. The syntax and semantics of CEL
  are documented at https://github.com/google/cel-spec. Example (Comparison):
  title: "Summary size limit" description: "Determines if a summary is less
  than 100 chars" expression: "document.summary.size() < 100" Example
  (Equality): title: "Requestor is owner" description: "Determines if
  requestor is the document owner" expression: "document.owner ==
  request.auth.claims.email" Example (Logic): title: "Public documents"
  description: "Determine whether the document should be publicly visible"
  expression: "document.type != 'private' && document.type != 'internal'"
  Example (Data Manipulation): title: "Notification string" description:
  "Create a notification string with a timestamp." expression: "'New message
  received at ' + string(document.create_time)" The exact variables and
  functions that may be referenced within an expression are determined by the
  service that evaluates it. See the service documentation for additional
  information.

  Fields:
    description: Optional. Description of the expression. This is a longer
      text which describes the expression, e.g. when hovered over it in a UI.
    expression: Textual representation of an expression in Common Expression
      Language syntax.
    location: Optional. String indicating the location of the expression for
      error reporting, e.g. a file name and a position in the file.
    title: Optional. Title for the expression, i.e. a short string describing
      its purpose. This can be used e.g. in UIs which allow to enter the
      expression.
  """

  description = _messages.StringField(1)
  expression = _messages.StringField(2)
  location = _messages.StringField(3)
  title = _messages.StringField(4)


class IamPoliciesCreatePolicyRequest(_messages.Message):
  r"""A IamPoliciesCreatePolicyRequest object.

  Fields:
    googleIamV2betaPolicy: A GoogleIamV2betaPolicy resource to be passed as
      the request body.
    parent: Required. The Cloud resource the new Policy is attached to. Takes
      the form: `policies/{attachment-point}/{kind-plural}`
    policyId: The ID to use for this policy, which will become the final
      component of the policy's resource name. Must match a-z{3,63}.
  """

  googleIamV2betaPolicy = _messages.MessageField('GoogleIamV2betaPolicy', 1)
  parent = _messages.StringField(2, required=True)
  policyId = _messages.StringField(3)


class IamPoliciesDeleteRequest(_messages.Message):
  r"""A IamPoliciesDeleteRequest object.

  Fields:
    etag: The expected etag of the policy to delete. If the policy was
      modified concurrently such that the etag changed, the delete operation
      will fail.
    name: Required. Resource name of the policy to delete.
  """

  etag = _messages.StringField(1)
  name = _messages.StringField(2, required=True)


class IamPoliciesGetRequest(_messages.Message):
  r"""A IamPoliciesGetRequest object.

  Fields:
    name: Required. Resource name of the policy to retrieve.
  """

  name = _messages.StringField(1, required=True)


class IamPoliciesListPoliciesRequest(_messages.Message):
  r"""A IamPoliciesListPoliciesRequest object.

  Fields:
    pageSize: The maximum number of policies to return. If unspecified, at
      most 1000 policies are returned. The maximum value is 1000; values above
      are 1000 truncated to 1000. The minimum value is 1000; values below 1000
      are increased to 1000.
    pageToken: A page token, received from a previous `ListPolicies` call.
      Provide this to retrieve the subsequent page.
    parent: Required. The Cloud resource that the policy is attached to. Takes
      the form: `policies/{attachment-point}/{kind-plural}`.
  """

  pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
  pageToken = _messages.StringField(2)
  parent = _messages.StringField(3, required=True)


class IamPoliciesOperationsGetRequest(_messages.Message):
  r"""A IamPoliciesOperationsGetRequest object.

  Fields:
    name: The name of the operation resource.
  """

  name = _messages.StringField(1, required=True)


class StandardQueryParameters(_messages.Message):
  r"""Query parameters accepted by all methods.

  Enums:
    FXgafvValueValuesEnum: V1 error format.
    AltValueValuesEnum: Data format for response.

  Fields:
    f__xgafv: V1 error format.
    access_token: OAuth access token.
    alt: Data format for response.
    callback: JSONP
    fields: Selector specifying which fields to include in a partial response.
    key: API key. Your API key identifies your project and provides you with
      API access, quota, and reports. Required unless you provide an OAuth 2.0
      token.
    oauth_token: OAuth 2.0 token for the current user.
    prettyPrint: Returns response with indentations and line breaks.
    quotaUser: Available to use for quota purposes for server-side
      applications. Can be any arbitrary string assigned to a user, but should
      not exceed 40 characters.
    trace: A tracing token of the form "token:<tokenid>" to include in api
      requests.
    uploadType: Legacy upload protocol for media (e.g. "media", "multipart").
    upload_protocol: Upload protocol for media (e.g. "raw", "multipart").
  """

  class AltValueValuesEnum(_messages.Enum):
    r"""Data format for response.

    Values:
      json: Responses with Content-Type of application/json
      media: Media download with context-dependent Content-Type
      proto: Responses with Content-Type of application/x-protobuf
    """
    json = 0
    media = 1
    proto = 2

  class FXgafvValueValuesEnum(_messages.Enum):
    r"""V1 error format.

    Values:
      _1: v1 error format
      _2: v2 error format
    """
    _1 = 0
    _2 = 1

  f__xgafv = _messages.EnumField('FXgafvValueValuesEnum', 1)
  access_token = _messages.StringField(2)
  alt = _messages.EnumField('AltValueValuesEnum', 3, default='json')
  callback = _messages.StringField(4)
  fields = _messages.StringField(5)
  key = _messages.StringField(6)
  oauth_token = _messages.StringField(7)
  prettyPrint = _messages.BooleanField(8, default=True)
  quotaUser = _messages.StringField(9)
  trace = _messages.StringField(10)
  uploadType = _messages.StringField(11)
  upload_protocol = _messages.StringField(12)


encoding.AddCustomJsonFieldMapping(
    StandardQueryParameters, 'f__xgafv', '$.xgafv')
encoding.AddCustomJsonEnumMapping(
    StandardQueryParameters.FXgafvValueValuesEnum, '_1', '1')
encoding.AddCustomJsonEnumMapping(
    StandardQueryParameters.FXgafvValueValuesEnum, '_2', '2')
