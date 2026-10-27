# Monitored single-GPU training

See [settings, monitoring, tmux operation and pending GPU verification](../../docs/lidar-monitored-training.md).

`configs/*-proposed.json` deliberately have both launch approvals disabled.
No command in preparation grants supervisor permission. Use `bash tmux.sh status
--config configs/align-proposed.json` to see that the stage has not started.

The new runner rejects historical schema-1 configs. Main training uses physical
GPU 2 only and never automatically starts another stage.
