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
"""Command for creating network firewall policy rules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import firewall_policy_rule_utils as rule_utils
from googlecloudsdk.api_lib.compute.network_firewall_policies import client
from googlecloudsdk.api_lib.compute.network_firewall_policies import region_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.network_firewall_policies import flags
from googlecloudsdk.command_lib.compute.network_firewall_policies import secure_tags_utils


class Create(base.CreateCommand):
  r"""Creates a Compute Engine network firewall policy rule.

  *{command}* is used to create network firewall policy rules.
  """

  NETWORK_FIREWALL_POLICY_ARG = None

  @classmethod
  def Args(cls, parser):
    cls.NETWORK_FIREWALL_POLICY_ARG = flags.NetworkFirewallPolicyRuleArgument(
        required=True, operation='create')
    cls.NETWORK_FIREWALL_POLICY_ARG.AddArgument(parser, operation_type='create')
    flags.AddAction(parser)
    flags.AddRulePriority(parser, operation='inserted')
    flags.AddSrcIpRanges(parser)
    flags.AddDestIpRanges(parser)
    flags.AddLayer4Configs(parser)
    flags.AddDirection(parser)
    flags.AddEnableLogging(parser)
    flags.AddDisabled(parser)
    flags.AddTargetServiceAccounts(parser)
    flags.AddDescription(parser)
    flags.AddSrcSecureTags(parser)
    flags.AddTargetSecureTags(parser)
    flags.AddSrcAddressGroups(parser)
    flags.AddDestAddressGroups(parser)
    if cls.ReleaseTrack() == base.ReleaseTrack.ALPHA:
      flags.AddSrcFqdns(parser)
      flags.AddDestFqdns(parser)
      flags.AddSrcRegionCodes(parser)
      flags.AddDestRegionCodes(parser)
    parser.display_info.AddCacheUpdater(flags.NetworkFirewallPoliciesCompleter)

  def Run(self, args):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    ref = self.NETWORK_FIREWALL_POLICY_ARG.ResolveAsResource(
        args, holder.resources)
    network_firewall_policy_rule_client = client.NetworkFirewallPolicyRule(
        ref=ref, compute_client=holder.client)
    if hasattr(ref, 'region'):
      network_firewall_policy_rule_client = region_client.RegionNetworkFirewallPolicyRule(
          ref, compute_client=holder.client)

    src_ip_ranges = []
    dest_ip_ranges = []
    layer4_configs = []
    target_service_accounts = []
    enable_logging = False
    disabled = False
    src_secure_tags = []
    target_secure_tags = []
    src_address_groups = []
    dest_address_groups = []
    src_fqdns = []
    dest_fqdns = []
    src_region_codes = []
    dest_region_codes = []
    if args.IsSpecified('src_ip_ranges'):
      src_ip_ranges = args.src_ip_ranges
    if args.IsSpecified('dest_ip_ranges'):
      dest_ip_ranges = args.dest_ip_ranges
    if args.IsSpecified('layer4_configs'):
      layer4_configs = args.layer4_configs
    if args.IsSpecified('target_service_accounts'):
      target_service_accounts = args.target_service_accounts
    if args.IsSpecified('enable_logging'):
      enable_logging = args.enable_logging
    if args.IsSpecified('disabled'):
      disabled = args.disabled
    if args.IsSpecified('src_secure_tags'):
      src_secure_tags = secure_tags_utils.TranslateSecureTagsForFirewallPolicy(
          holder.client, args.src_secure_tags)
    if args.IsSpecified('target_secure_tags'):
      target_secure_tags = secure_tags_utils.TranslateSecureTagsForFirewallPolicy(
          holder.client, args.target_secure_tags)
    if args.IsSpecified('src_address_groups'):
      src_address_groups = args.src_address_groups
    if args.IsSpecified('dest_address_groups'):
      dest_address_groups = args.dest_address_groups
    if self.ReleaseTrack() == base.ReleaseTrack.ALPHA:
      if args.IsSpecified('src_fqdns'):
        src_fqdns = args.src_fqdns
      if args.IsSpecified('dest_fqdns'):
        dest_fqdns = args.dest_fqdns
      if args.IsSpecified('src_region_codes'):
        src_region_codes = args.src_region_codes
      if args.IsSpecified('dest_region_codes'):
        dest_region_codes = args.dest_region_codes
    layer4_config_list = rule_utils.ParseLayer4Configs(layer4_configs,
                                                       holder.client.messages)
    if self.ReleaseTrack() == base.ReleaseTrack.ALPHA:
      matcher = holder.client.messages.FirewallPolicyRuleMatcher(
          srcIpRanges=src_ip_ranges,
          destIpRanges=dest_ip_ranges,
          layer4Configs=layer4_config_list,
          srcSecureTags=src_secure_tags,
          srcAddressGroups=src_address_groups,
          destAddressGroups=dest_address_groups,
          srcFqdns=src_fqdns,
          destFqdns=dest_fqdns,
          srcRegionCodes=src_region_codes,
          destRegionCodes=dest_region_codes)
    else:
      matcher = holder.client.messages.FirewallPolicyRuleMatcher(
          srcIpRanges=src_ip_ranges,
          destIpRanges=dest_ip_ranges,
          layer4Configs=layer4_config_list,
          srcSecureTags=src_secure_tags,
          srcAddressGroups=src_address_groups,
          destAddressGroups=dest_address_groups)
    traffic_direct = (
        holder.client.messages.FirewallPolicyRule.DirectionValueValuesEnum
        .INGRESS)
    if args.IsSpecified('direction'):
      if args.direction == 'INGRESS':
        traffic_direct = (
            holder.client.messages.FirewallPolicyRule.DirectionValueValuesEnum
            .INGRESS)
      else:
        traffic_direct = (
            holder.client.messages.FirewallPolicyRule.DirectionValueValuesEnum
            .EGRESS)

    firewall_policy_rule = holder.client.messages.FirewallPolicyRule(
        priority=rule_utils.ConvertPriorityToInt(args.priority),
        action=args.action,
        match=matcher,
        direction=traffic_direct,
        targetServiceAccounts=target_service_accounts,
        description=args.description,
        enableLogging=enable_logging,
        disabled=disabled,
        targetSecureTags=target_secure_tags)

    return network_firewall_policy_rule_client.Create(
        firewall_policy=args.firewall_policy,
        firewall_policy_rule=firewall_policy_rule)


Create.detailed_help = {
    'EXAMPLES':
        """\
    To create a rule with priority ``10'' in a global network firewall policy
    with name ``my-policy'' and description ``example rule'', run:

        $ {command} 10 --firewall-policy=my-policy --action=allow --description="example rule" --global-firewall-policy

    To create a rule with priority ``10'' in a regional network firewall policy
    with name ``my-region-policy'' and description ``example rule'', in
    region ``region-a'', run:

        $ {command} 10 --firewall-policy=my-policy --action=allow --description="example rule"
    """,
}
