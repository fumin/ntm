Neural Turing Machines
-----
Package ntm implements the Neural Turing Machine architecture as described in A.Graves, G. Wayne, and I. Danihelka. arXiv preprint arXiv:1410.5401, 2014.

Using this package along its subpackages, the "copy", "repeatcopy" and "ngram" tasks mentioned in the paper were verified.
For each of these tasks, the successfully trained models are saved under the filenames "seedA_B",
where A is the number indicating the seed provided to rand.Seed in the training process, and B is the iteration number in which the trained weights converged.

## Reproducing results in the paper
The following sections detail the steps of verifying the results in the paper. All commands are assumed to be run in the $GOPATH/github.com/fumin/ntm folder.

### Copy
#### Train
To start training, run `go run copytask/train/main.go` which not only commences training but also starts a web server that would be convenient to track progress.
To print debug information about the training process, run `curl http://localhost:8088/PrintDebug`.
To track the cross-entropy loss during the training process, run `curl http://localhost:8088/Loss`.
To save the trained weights to disk, run `curl http://localhost:8088/Weights > weights`.
#### Testing
To test the saved weights in the previous training step, run `go run copytask/test/main.go -weightsFile=weights`. Alternatively, you can also specify one of the successfully trained weights in the copytask/test folder such as the file copytask/test/seed11_28000.
Upon running the above command, a web server would be started which can be accessed at http://localhost:9000/.

### Repeat copy
To experiment on the repeat copy task, follow the steps of the copy task except changing the package from `copytask` to `repeatcopy`.

### Dynamic N-grams
To experiment on the dynamic n-grams task, follow the steps of the copy task except changing the package from `copytask` to `ngram`.

## Testing
To run the tests of this package, run `go test -test.v`.
