#include <windows.h>

int main() {
    // 等待0.5秒（500毫秒）
    Sleep(500);

    // 关闭显示器
    SendMessage(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, (LPARAM)2);

    return 0;
}
