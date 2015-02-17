prof:
	go get github.com/fumin/ntm/copytask/train
	${GOPATH}/bin/train -cpuprofile=train.prof
	go tool pprof ${GOPATH}/bin/train train.prof

clean:
	rm -f train.prof
