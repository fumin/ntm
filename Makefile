SCP=scp -i ~/cardinalblue/cbauthenticator/config/certs/aws_ec2_piccollage.pem
REMOTE=ec2-user@54.173.101.231:/home/ec2-user/gopath/src/github.com/fumin/ntm

prof:
	go get github.com/fumin/ntm/copy
	${GOPATH}/bin/copy -cpuprofile=copy.prof
	go tool pprof ${GOPATH}/bin/copy copy.prof

scp:
	${SCP} Makefile ntm.go math.go ${REMOTE}/
	${SCP} copy/main.go ${REMOTE}/copy/

clean:
	rm -f copy.prof
