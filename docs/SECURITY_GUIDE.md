# Security guide

[SECURITY.md](../SECURITY.md) is the authoritative policy and vulnerability
reporting document. The compatibility secret loader remains source-visible;
this page does not certify or prescribe a production secrets deployment.

ELVIS is experimental software, not an enterprise-certified system. It has no
SOC 2, FIPS, OWASP, or similar certification. A Vault/OpenBao deployment may
provide controls, but running Vault does not certify this application.

## Minimum local practice

- Use Python 3.14 and a disposable paper environment.
- Keep passwords, API keys, libpq service files, pgpass files, and receipts
  outside Git.
- Pass secrets through a reviewed secret store or external files, never
  command arguments or release bundles.
- Give tokens the minimum permissions and rotate them after exposure.
- Treat the root Compose credentials as compatibility development defaults,
  not production secrets.
- Keep live trading disabled; `ACTIVE` remains a **NO-GO**.

For the V2 preview, follow [INSTALL_V2.md](../INSTALL_V2.md) and the selected
operator runbook. A successful health check, bootstrap, or receipt is not a
security audit or cut-over approval.
