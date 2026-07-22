# GitHub Actions Workflows

See the GitHub documentation for more information on GitHub Actions in general.

## Notes & Security Best Practices

* **Action Pinning**: Per Google Open Source security policy, non-Google
  GitHub Actions should reference pinned versions or releases. We use
  [Ratchet](https://github.com/sethvargo/ratchet) to pin specific versions. If
  you'd like to update or add an action, you can write something like
  `uses: actions/checkout@v4`, and then run
  `ratchet pin .github/workflows/*.yml` to convert mutable tags to immutable
  commit hashes with version comments.
* **Fork PR Protection**: Self-hosted Cloud TPU VM runners
  (`linux-x86-ct6e-180-8tpu`) execute on Google Cloud VPC infrastructure. To
  prevent untrusted code execution from arbitrary forks, all TPU workflows MUST
  include the security guard:
  ```yaml
  if: github.event.repository.fork == false && github.repository_owner == 'jax-ml'
  ```
  Fork PRs do not execute TPU tests automatically. When testing an external
  contributor's PR on TPU, a maintainer must trigger a manual run or test
  internally.
