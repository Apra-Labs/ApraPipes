package main

import (
	"net/http"
	"bytes"
	"os"
	"errors"
	"net"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

//Message resp struct
// type Message struct {
// 	Status  int         `json:"status"`
// 	Payload interface{} `json:"payload"`
// }


func main() {
	HTTPAPIServer()
}
//HTTPAPIServer start http server routes - just for upgrading http to ws
func HTTPAPIServer() {
	var public *gin.Engine
	gin.SetMode(gin.ReleaseMode)
	public = gin.New()

	public.Use(CrossOrigin())
	//WebRTCSignalling
	public.GET("/signalling", func(c *gin.Context) {
		WebRTCSignallingServer(c.Writer, c.Request)
	})
	logrus.Printf("Running webrtc signalling server on localhost:8083/signalling")
	err := public.Run("localhost:8083")
	if err != nil {
		logrus.Println(err.Error())
		os.Exit(1)
	}
}
func CrossOrigin() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	}
}
//Signalling server logic starts here
type Peer struct {
	PeerID string
	Conn   *websocket.Conn
	Status string
}

var peers = make(map[string]Peer)
var sessions = make(map[string]string)

var wsupgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

var (
	newline = []byte{'\n'}
	space   = []byte{' '}
)

func helloPeer(conn *websocket.Conn) string {
	raddr := conn.RemoteAddr()
	msgType, hello, err := conn.ReadMessage()
	logrus.Printf("%d is the message type", msgType)
	if err != nil {
		if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
			logrus.Printf("error: %v", err)
		}
	}
	hello = bytes.TrimSpace(bytes.Replace(hello, newline, space, -1))
	helloSplit := strings.Split(string(hello), " ")
	if len(helloSplit) != 2 {
		conn.Close()
		logrus.Printf("Invalid protocol from %s", raddr)
	}
	if helloSplit[0] != "HELLO" {
		conn.Close()
		logrus.Printf("Invalid hello from %s", raddr)
	}
	if _, ok := peers[helloSplit[1]]; ok {
		conn.Close()
		logrus.Printf("Invalid uid %s, already present", helloSplit[1])
	}
	conn.WriteMessage(1, []byte("HELLO"))
	peerID := helloSplit[1]
	return peerID
}

func cleanUPSession(peerID string) {
	if _, ok := sessions[peerID]; ok {
		otherPeerID := sessions[peerID]
		delete(sessions, peerID)
		logrus.Printf("Cleaned up %s session", peerID)
		if _, ok := sessions[otherPeerID]; ok {
			delete(sessions,otherPeerID)
			logrus.Printf("Also cleaned up %s session", otherPeerID)
			if _, ok := peers[otherPeerID]; ok {
				logrus.Printf("Closing connection to %s", otherPeerID)
				peers[otherPeerID].Conn.Close()
				delete(peers, otherPeerID)
			}
		}
	}
}

func removePeer(peerID string) {
	cleanUPSession(peerID)
	if _,ok := peers[peerID]; ok {
		peers[peerID].Conn.Close()
		delete(peers,peerID)
		logrus.Printf("Disconnected from peer %s",peerID)
	}
}

func receiveMessage(conn *websocket.Conn, raddr net.Addr) string {
	msg := ""
	for msg == "" {
		_, msgBytes, err := conn.ReadMessage()
		if err != nil {
				return "ConnTerminatedByClient"
		}
		msgBytes = bytes.TrimSpace(bytes.Replace(msgBytes, newline, space, -1))
		msg = string(msgBytes)
	}
	return msg
}

func connectionHandler(peerID string, conn *websocket.Conn) {
	raddr := conn.RemoteAddr()
	var peerStatus string
	peers[peerID] = Peer{PeerID: peerID, Conn: conn, Status: peerStatus}
	logrus.Printf("Registered peer %s at %s",peerID,raddr)
	for {
		msg := receiveMessage(conn, raddr)
		peerStatus = peers[peerID].Status
		if msg == "ConnTerminatedByClient" {
			return
		} else if peerStatus != "" {
			if peerStatus == "session" {
				otherID := sessions[peerID]
				logrus.Printf("%s -> %s : %s", peerID, otherID, msg)
				peers[otherID].Conn.WriteMessage(1, []byte(msg))
			} else {
				logrus.Printf("Unknown peer status %s", peerStatus)
			}
		} else if strings.Contains(msg, "SESSION") {
			logrus.Printf("%s command %s", peerID, msg)
			msgSplit := strings.Split(msg, " ")
			calleeID := msgSplit[1]
			if _, ok := peers[calleeID]; !ok {
				conn.WriteMessage(1, []byte("ERROR peer "+calleeID+" not found"))
				continue
			}
			if peerStatus != "" {
				conn.WriteMessage(1, []byte("ERROR peer"+calleeID+" busy"))
				continue
			}
			wsc := peers[calleeID].Conn
			wsc.WriteMessage(1, []byte("SESSION_OK"))
			logrus.Printf("Session from %s (%s) to %s (%s)", peerID, raddr, calleeID, wsc.RemoteAddr())
			tempPeer := peers[peerID]
			tempPeer.Status = "session"
			peers[peerID] = tempPeer
			peerStatus = "session"

			sessions[peerID] = calleeID

			tempPeer = peers[calleeID]
			tempPeer.Status = "session"
			peers[calleeID] = tempPeer

			sessions[calleeID] = peerID
		} else {
			logrus.Printf("Ignoring unknown message %s from %s", msg, peerID)
		}
	}
}

func WebRTCSignallingServer(w http.ResponseWriter, r *http.Request) {
	wsupgrader.CheckOrigin = func(r *http.Request) bool { return true }
	conn, err := wsupgrader.Upgrade(w, r, nil)
	if err != nil {
		logrus.Println(err)
		return
	}
	logrus.Printf("Connected to %s", conn.RemoteAddr())
	peerId := helloPeer(conn)
	conn.SetCloseHandler(func(code int, text string) error {
		removePeer(peerId)
		return errors.New("ClientClosedConnection")
	})
	connectionHandler(peerId, conn)
}