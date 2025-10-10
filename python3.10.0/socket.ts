import { Server, Socket } from "socket.io";
import jwt from "jsonwebtoken";
import http from "http";

const SECRET_KEY = process.env.JWT_SECRET || "your-secret-key";
const MAX_DEFAULT_CONNECTIONS = 5;

interface UserInfo {
  role: string;
  room: string;
  username?: string | undefined;
  seatIndex?: number | undefined;
}

interface RoomInfo {
  maxClients: number;
  occupiedSeats: Set<number>;
  seatAssignments: Map<number, string>; // seatIndex -> socketId
  recipeId?: string | undefined; // 選択されたレシピID
  sessionActive?: boolean; // セッションの状態
}

interface StudentProgress {
  currentStep: number;
  recipeId: string;
  lastUpdate: string;
}

function setupSocket(server: http.Server): void {
  const io = new Server(server, {
    allowEIO3: true,
    cors: {
      origin: "*",
      methods: ["GET", "POST"],
      credentials: true,
    },
  });

  // middleware
  io.use((socket, next) => {
    const token = socket.handshake.auth?.token as string;
    if (!token) {
      return next(new Error("トークンが必要です"));
    }

    try {
      const payload = jwt.verify(token, SECRET_KEY);
      (socket as any).user = payload;
      next();
    } catch (err) {
      return next(new Error("無効なトークンです"));
    }
  });

  // データ管理
  const userMap = new Map<string, UserInfo>();
  const roomMap = new Map<string, RoomInfo>();
  const progressMap = new Map<string, StudentProgress>(); // userId -> progress

  io.on("connection", (socket: Socket) => {
    console.log(`A user(${socket.id}) connected`);

    // ルーム情報取得イベント
    socket.on("getRoomInfo", (data: { room: string }) => {
      const { room } = data;
      const roomInfo = roomMap.get(room);

      if (!roomInfo) {
        socket.emit("roomInfo", {
          success: false,
          message: "ルームが見つかりません",
        });
        return;
      }

      socket.emit("roomInfo", {
        success: true,
        maxSeats: roomInfo.maxClients - 1, // ホストを除いた座席数
        occupiedSeats: Array.from(roomInfo.occupiedSeats),
      });
    });

    // ルーム参加
    socket.on(
      "join",
      async (data: {
        role: string;
        room: string;
        username?: string;
        maxClients?: number;
        seatIndex?: number;
        recipeId?: string;
      }) => {
        const { role, room, maxClients, username, seatIndex, recipeId } = data;

        if (role === "host") {
          if (roomMap.has(room)) {
            console.log("Room already exists, overwriting:", room);
            roomMap.delete(room);
          }

          // ホスト用のルーム情報を作成
          roomMap.set(room, {
            maxClients: (maxClients || MAX_DEFAULT_CONNECTIONS) + 1,
            occupiedSeats: new Set<number>(),
            seatAssignments: new Map<number, string>(),
            recipeId: recipeId,
            sessionActive: false,
          });

          socket.join(room);
          userMap.set(socket.id, {
            role,
            room,
            username: username || undefined,
          });

          console.log(`Host created room: ${room} with recipe: ${recipeId}`);
          return;
        }

        // 生徒の場合の処理
        const roomInfo = roomMap.get(room);
        if (!roomInfo) {
          socket.emit("message", { type: "room_not_found" });
          return;
        }

        const currentRoomSize = io.sockets.adapter.rooms.get(room)?.size || 0;
        if (currentRoomSize >= roomInfo.maxClients) {
          socket.emit("message", { type: "room_full" });
          return;
        }

        // 座席指定がある場合の処理
        if (seatIndex !== undefined) {
          const maxSeats = roomInfo.maxClients - 1;
          if (seatIndex < 0 || seatIndex >= maxSeats) {
            socket.emit("message", {
              type: "invalid_seat",
              message: "無効な座席番号です",
            });
            return;
          }

          if (roomInfo.occupiedSeats.has(seatIndex)) {
            socket.emit("message", { type: "seat_occupied" });
            return;
          }

          // 座席を予約
          roomInfo.occupiedSeats.add(seatIndex);
          roomInfo.seatAssignments.set(seatIndex, socket.id);
        }

        socket.join(room);
        userMap.set(socket.id, {
          role,
          room,
          username,
          seatIndex,
        });

        // ルーム内の全員に参加通知
        io.to(room).emit("message", {
          type: "student_joined",
          status: "joined",
          userId: socket.id,
          role: role,
          room: room,
          username: username,
          seatIndex: seatIndex,
        });

        // 座席状況の更新を全員に通知
        io.to(room).emit("seatUpdate", {
          occupiedSeats: Array.from(roomInfo.occupiedSeats),
        });

        console.log(
          `Student ${username} joined room ${room} at seat ${seatIndex}`
        );
      }
    );

    // セッション開始
    socket.on("startSession", (data: { room: string; recipeId: string }) => {
      const user = userMap.get(socket.id);
      if (!user || user.role !== "host") {
        return;
      }

      const roomInfo = roomMap.get(data.room);
      if (!roomInfo) {
        return;
      }

      roomInfo.sessionActive = true;
      roomInfo.recipeId = data.recipeId;

      console.log(
        `Session started in room ${data.room} with recipe ${data.recipeId}`
      );

      // 生徒側にセッション開始通知
      io.to(data.room).emit("sessionStarted", {
        recipeId: data.recipeId,
        room: data.room,
      });
    });

    // セッション終了
    socket.on("endSession", (data: { room: string }) => {
      const user = userMap.get(socket.id);
      if (!user || user.role !== "host") {
        return;
      }

      const roomInfo = roomMap.get(data.room);
      if (!roomInfo) {
        return;
      }

      roomInfo.sessionActive = false;

      console.log(`Session ended in room ${data.room}`);

      // 生徒側にセッション終了通知
      io.to(data.room).emit("sessionEnded", {
        room: data.room,
      });
    });

    // セッション応答（生徒からの確認）
    socket.on(
      "sessionResponse",
      (data: {
        room: string;
        userId: string;
        type: string;
        status: string;
      }) => {
        console.log(
          `Session response from ${data.userId}: ${data.type} - ${data.status}`
        );

        // 教師側に応答を転送
        const room = data.room;
        const clients = Array.from(
          io.sockets.adapter.rooms.get(room) ?? []
        ) as string[];

        clients.forEach((clientId) => {
          const user = userMap.get(clientId);
          if (user && user.role === "host") {
            io.to(clientId).emit("sessionResponse", {
              userId: data.userId,
              type: data.type,
              status: data.status,
              timeStamp: new Date().toISOString(),
            });
          }
        });
      }
    );

    // 生徒の進捗更新
    socket.on(
      "studentProgress",
      (data: {
        room: string;
        userId: string;
        username: string;
        seatIndex: number;
        currentStep: number;
        recipeId: string;
      }) => {
        console.log("Student progress update:", data);

        // 進捗情報を保存
        progressMap.set(data.userId, {
          currentStep: data.currentStep,
          recipeId: data.recipeId,
          lastUpdate: new Date().toISOString(),
        });

        // 教師側に進捗を通知
        const room = data.room;
        const clients = Array.from(
          io.sockets.adapter.rooms.get(room) ?? []
        ) as string[];

        clients.forEach((clientId) => {
          const user = userMap.get(clientId);
          if (user && user.role === "host") {
            io.to(clientId).emit("studentProgress", {
              userId: data.userId,
              username: data.username,
              seatIndex: data.seatIndex,
              currentStep: data.currentStep,
              recipeId: data.recipeId,
              timeStamp: new Date().toISOString(),
            });
          }
        });
      }
    );

    // 危険通知
    socket.on(
      "dangerAlert",
      (data: {
        room: string;
        userId: string;
        username: string;
        seatIndex: number;
        message?: string;
      }) => {
        console.log("Danger alert received:", data);

        // 教師側に危険通知を送信
        const room = data.room;
        const clients = Array.from(
          io.sockets.adapter.rooms.get(room) ?? []
        ) as string[];

        clients.forEach((clientId) => {
          const user = userMap.get(clientId);
          if (user && user.role === "host") {
            io.to(clientId).emit("dangerAlert", {
              userId: data.userId,
              username: data.username,
              seatIndex: data.seatIndex,
              message: data.message || "危険な状況が発生しました",
              timeStamp: new Date().toISOString(),
            });
          }
        });
      }
    );

    // 既存のメッセージ送信機能
    socket.on(
      "sendJson",
      (data: {
        room: string;
        userId: string;
        message: any;
        payload?: any;
        seatIndex?: number;
      }) => {
        console.log("Received JSON data:", {
          room: data.room,
          userId: data.userId,
          message: data.message,
          seatIndex: data.seatIndex,
        });

        const room = data.room;
        const clients = Array.from(
          io.sockets.adapter.rooms.get(room) ?? []
        ) as string[];

        clients.forEach((clientId) => {
          const user = userMap.get(clientId);
          if (user && user.role === "host") {
            io.to(clientId).emit("receiveJson", {
              message: data.message,
              payload: data.payload,
              room: data.room,
              userId: data.userId,
              seatIndex: data.seatIndex,
              timeStamp: new Date().toISOString(),
            });
          }
        });
      }
    );

    // 切断処理
    socket.on("disconnect", () => {
      console.log(`A user(${socket.id}) disconnected`);
      const user = userMap.get(socket.id);
      if (user) {
        // 座席の解放
        const roomInfo = roomMap.get(user.room);
        if (roomInfo && user.seatIndex !== undefined) {
          roomInfo.occupiedSeats.delete(user.seatIndex);
          roomInfo.seatAssignments.delete(user.seatIndex);

          // 座席状況の更新を通知
          io.to(user.room).emit("seatUpdate", {
            occupiedSeats: Array.from(roomInfo.occupiedSeats),
          });
        }

        // 進捗情報を削除
        progressMap.delete(socket.id);

        io.to(user.room).emit("message", {
          type: "student_left",
          status: "left",
          userId: socket.id,
          role: user.role,
          room: user.room,
          username: user.username,
          seatIndex: user.seatIndex,
        });

        userMap.delete(socket.id);
        socket.leave(user.room);

        console.log(
          `User ${user.username} left room ${user.room} from seat ${user.seatIndex}`
        );
      }
    });

    // ホストによるルーム閉鎖
    socket.on("closeRoom", (data: { room: string }) => {
      const host = userMap.get(socket.id);
      if (host && host.role === "host") {
        const room = data.room || host.room;
        roomMap.delete(room);

        // 該当ルームの進捗情報をクリア
        const roomClients = Array.from(
          io.sockets.adapter.rooms.get(room) ?? []
        ) as string[];
        roomClients.forEach((clientId) => {
          progressMap.delete(clientId);
        });

        console.log(`Room ${room} closed by host`);
        io.to(room).emit("message", { message: "ホスト切断通知" });
        io.in(room).disconnectSockets(true);
      }
    });

    // 座席変更（既存機能）
    socket.on("changeSeat", (data: { room: string; newSeatIndex: number }) => {
      const user = userMap.get(socket.id);
      if (!user || user.role !== "student") {
        socket.emit("message", {
          type: "error",
          message: "座席変更権限がありません",
        });
        return;
      }

      const roomInfo = roomMap.get(data.room);
      if (!roomInfo) {
        socket.emit("message", { type: "room_not_found" });
        return;
      }

      const maxSeats = roomInfo.maxClients - 1;
      if (data.newSeatIndex < 0 || data.newSeatIndex >= maxSeats) {
        socket.emit("message", {
          type: "invalid_seat",
          message: "無効な座席番号です",
        });
        return;
      }

      if (roomInfo.occupiedSeats.has(data.newSeatIndex)) {
        socket.emit("message", { type: "seat_occupied" });
        return;
      }

      // 古い座席を解放
      if (user.seatIndex !== undefined) {
        roomInfo.occupiedSeats.delete(user.seatIndex);
        roomInfo.seatAssignments.delete(user.seatIndex);
      }

      // 新しい座席を予約
      roomInfo.occupiedSeats.add(data.newSeatIndex);
      roomInfo.seatAssignments.set(data.newSeatIndex, socket.id);
      user.seatIndex = data.newSeatIndex;
      userMap.set(socket.id, user);

      // 座席変更通知
      io.to(data.room).emit("message", {
        type: "seat_changed",
        userId: socket.id,
        username: user.username,
        oldSeatIndex: user.seatIndex,
        newSeatIndex: data.newSeatIndex,
      });

      // 座席状況更新
      io.to(data.room).emit("seatUpdate", {
        occupiedSeats: Array.from(roomInfo.occupiedSeats),
      });

      console.log(`User ${user.username} changed seat to ${data.newSeatIndex}`);
    });

    // ルーム退出
    socket.on("leave-room", (data: { roomName: string }) => {
      leaveRoom(socket, data.roomName);
    });

    // デバッグ用：ルーム状況確認
    socket.on("getRoomStatus", (data: { room: string }) => {
      const user = userMap.get(socket.id);
      if (!user || user.role !== "host") {
        return;
      }

      const roomInfo = roomMap.get(data.room);
      if (roomInfo) {
        socket.emit("roomStatus", {
          room: data.room,
          maxClients: roomInfo.maxClients,
          occupiedSeats: Array.from(roomInfo.occupiedSeats),
          seatAssignments: Object.fromEntries(roomInfo.seatAssignments),
          recipeId: roomInfo.recipeId,
          sessionActive: roomInfo.sessionActive,
        });
      }
    });

    function leaveRoom(socket: Socket, roomName: string) {
      const user = userMap.get(socket.id);
      if (!user || user.room !== roomName) {
        console.log(
          `ユーザー(${socket.id})はルーム(${roomName})に所属していません`
        );
        return;
      }

      const roomData = io.sockets.adapter.rooms.get(roomName);
      if (!roomData) {
        console.log(`ルーム(${roomName})は存在しません`);
        return;
      }

      // 座席の解放
      const roomInfo = roomMap.get(roomName);
      if (roomInfo && user.seatIndex !== undefined) {
        roomInfo.occupiedSeats.delete(user.seatIndex);
        roomInfo.seatAssignments.delete(user.seatIndex);

        // 座席状況の更新を通知
        io.to(roomName).emit("seatUpdate", {
          occupiedSeats: Array.from(roomInfo.occupiedSeats),
        });
      }

      // 進捗情報を削除
      progressMap.delete(socket.id);

      // ユーザーをルームから削除
      userMap.delete(socket.id);
      socket.leave(roomName);

      // ルームが空になった場合、ルームデータを削除
      if (roomData.size === 0) {
        roomMap.delete(roomName);
        console.log(`ルーム(${roomName})が空になったため削除されました`);
      }

      // 他のクライアントに通知
      io.to(roomName).emit("message", {
        type: "student_left",
        status: "left",
        userId: socket.id,
        role: user.role,
        room: roomName,
        username: user.username,
        seatIndex: user.seatIndex,
      });

      console.log(`ユーザー(${socket.id})がルーム(${roomName})を退出しました`);
    }
  });
}

export { setupSocket };
