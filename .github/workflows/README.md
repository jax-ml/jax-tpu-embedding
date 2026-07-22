# GitHub Actions Workflows

See the GitHub documentation for more information on GitHub Actions in general.

## Notes & Security Best Practices

* **Action Pinning**: Per Google Open Source security policy, non-Google GitHub
  Actions should reference pinned versions or releases. We use Ratchet
  (`github.com/sethvarinternal link:ratchet`) to pin specific versions. If you'd like to
  update or add an action, you can write something like `uses:
  actions/checkout@v4`, and then run `ratchet pin .github/workflows/*.yml` to
  convert mutable tags to immutable commit hashes with version comments.
