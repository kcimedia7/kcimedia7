"""Quiet COLMAP's glog output so test failures are readable."""
import os

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")
