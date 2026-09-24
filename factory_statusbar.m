#import <Cocoa/Cocoa.h>
#import <sys/file.h>
#import <sys/stat.h>

@interface StatusBarController : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSURL *stateDirectory;
@property(nonatomic, copy) NSString *pythonPath;
@property(nonatomic, copy) NSString *scriptPath;
@property(nonatomic, strong) NSStatusItem *statusItem;
@property(nonatomic, strong) NSMenuItem *nextItem;
@property(nonatomic, strong) NSMenuItem *countdownItem;
@property(nonatomic, strong) NSMenuItem *publishNowItem;
@property(nonatomic, strong) NSMenuItem *postingToggleItem;
@property(nonatomic, strong) NSMenuItem *retryItem;
@property(nonatomic, strong) NSTimer *timer;
@property(nonatomic, copy) NSString *currentStatus;
@property(nonatomic) BOOL postingEnabled;
@end

@implementation StatusBarController

- (instancetype)initWithStateDirectory:(NSURL *)stateDirectory
                             pythonPath:(NSString *)pythonPath
                              scriptPath:(NSString *)scriptPath {
    self = [super init];
    if (self) {
        _stateDirectory = stateDirectory;
        _pythonPath = [pythonPath copy];
        _scriptPath = [scriptPath copy];
        _currentStatus = @"waiting";
        _postingEnabled = YES;
    }
    return self;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    self.statusItem = [[NSStatusBar systemStatusBar] statusItemWithLength:NSVariableStatusItemLength];
    NSStatusBarButton *button = self.statusItem.button;
    button.image = [NSImage imageWithSystemSymbolName:@"paperplane.fill"
                            accessibilityDescription:@"Factory Autoposter"];
    button.imagePosition = NSImageLeading;
    button.title = @" --:--";

    NSMenu *menu = [[NSMenu alloc] init];
    self.nextItem = [[NSMenuItem alloc] initWithTitle:@"Следующая комбинация: —"
                                               action:nil keyEquivalent:@""];
    self.countdownItem = [[NSMenuItem alloc] initWithTitle:@"До публикации: —"
                                                    action:nil keyEquivalent:@""];
    [menu addItem:self.nextItem];
    [menu addItem:self.countdownItem];
    [menu addItem:[NSMenuItem separatorItem]];

    self.publishNowItem = [[NSMenuItem alloc] initWithTitle:@"Опубликовать сейчас"
                                                     action:@selector(publishNow:)
                                              keyEquivalent:@""];
    self.publishNowItem.target = self;
    [menu addItem:self.publishNowItem];

    self.postingToggleItem = [[NSMenuItem alloc] initWithTitle:@"Выключить автопостинг"
                                                        action:@selector(togglePosting:)
                                                 keyEquivalent:@""];
    self.postingToggleItem.target = self;
    [menu addItem:self.postingToggleItem];

    self.retryItem = [[NSMenuItem alloc] initWithTitle:@"Повторить через..."
                                                action:@selector(chooseRetryDelay:)
                                         keyEquivalent:@""];
    self.retryItem.target = self;
    self.retryItem.enabled = NO;
    [menu addItem:self.retryItem];

    NSMenuItem *tableItem = [[NSMenuItem alloc] initWithTitle:@"Открыть таблицу комбинаций"
                                                       action:@selector(openTable:)
                                                keyEquivalent:@""];
    tableItem.target = self;
    [menu addItem:tableItem];
    self.statusItem.menu = menu;

    [self refresh];
    self.timer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                  target:self
                                                selector:@selector(refresh)
                                                userInfo:nil
                                                 repeats:YES];
}

- (NSDictionary *)loadProgress {
    NSURL *path = [self.stateDirectory URLByAppendingPathComponent:@"progress.json"];
    NSData *data = [NSData dataWithContentsOfURL:path];
    if (!data) return nil;
    id payload = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    return [payload isKindOfClass:[NSDictionary class]] ? payload : nil;
}

- (BOOL)loadPostingEnabled {
    NSURL *path = [self.stateDirectory URLByAppendingPathComponent:@"control.json"];
    NSData *data = [NSData dataWithContentsOfURL:path];
    if (!data) return YES;
    NSDictionary *payload = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    NSNumber *value = [payload isKindOfClass:[NSDictionary class]] ? payload[@"posting_enabled"] : nil;
    return value ? value.boolValue : YES;
}

- (NSInteger)remainingSeconds:(NSString *)value {
    if (![value isKindOfClass:[NSString class]] || value.length == 0) return -1;
    NSISO8601DateFormatter *formatter = [[NSISO8601DateFormatter alloc] init];
    NSDate *date = [formatter dateFromString:value];
    if (!date) return -1;
    return MAX(0, (NSInteger)[date timeIntervalSinceNow]);
}

- (NSString *)fullDuration:(NSInteger)seconds {
    return [NSString stringWithFormat:@"%02ld:%02ld:%02ld",
            (long)(seconds / 3600), (long)((seconds % 3600) / 60), (long)(seconds % 60)];
}

- (NSString *)shortDuration:(NSInteger)seconds {
    if (seconds >= 3600) {
        return [NSString stringWithFormat:@"%ld:%02ld",
                (long)(seconds / 3600), (long)((seconds % 3600) / 60)];
    }
    return [NSString stringWithFormat:@"%02ld:%02ld",
            (long)(seconds / 60), (long)(seconds % 60)];
}

- (void)refresh {
    NSDictionary *payload = [self loadProgress];
    if (!payload) return;
    self.currentStatus = [payload[@"status"] isKindOfClass:[NSString class]]
        ? payload[@"status"] : @"waiting";
    NSString *title = [payload[@"title"] isKindOfClass:[NSString class]]
        ? payload[@"title"] : @"Factory Autoposter";
    NSString *detail = [payload[@"detail"] isKindOfClass:[NSString class]]
        ? payload[@"detail"] : @"";
    NSInteger seconds = [self remainingSeconds:payload[@"next_run_at"]];
    self.postingEnabled = [self loadPostingEnabled];
    self.nextItem.title = title;

    if ([self.currentStatus isEqualToString:@"running"]) {
        NSInteger percent = [payload[@"percent"] integerValue];
        self.countdownItem.title = [NSString stringWithFormat:@"Прогресс: %ld%% · %@",
                                    (long)percent, detail];
        self.statusItem.button.title = [NSString stringWithFormat:@" %ld%%", (long)percent];
    } else if (seconds >= 0) {
        self.countdownItem.title = [NSString stringWithFormat:@"До публикации: %@",
                                    [self fullDuration:seconds]];
        self.statusItem.button.title = [NSString stringWithFormat:@" %@",
                                        [self shortDuration:seconds]];
    } else {
        self.countdownItem.title = detail.length ? detail : @"Ожидание";
        self.statusItem.button.title = @"";
    }
    self.publishNowItem.enabled = self.postingEnabled && [self.currentStatus isEqualToString:@"waiting"];
    self.postingToggleItem.title = self.postingEnabled
        ? @"Выключить автопостинг" : @"Включить автопостинг";
    self.retryItem.enabled = [self.currentStatus isEqualToString:@"error"];
}

- (void)publishNow:(id)sender {
    if (![self.currentStatus isEqualToString:@"waiting"]) return;
    self.publishNowItem.enabled = NO;
    NSTask *task = [[NSTask alloc] init];
    task.executableURL = [NSURL fileURLWithPath:self.pythonPath];
    task.arguments = @[
        self.scriptPath,
        @"--state-dir", self.stateDirectory.path,
        @"publish-now",
    ];
    task.standardOutput = [NSFileHandle fileHandleWithNullDevice];
    task.standardError = [NSFileHandle fileHandleWithNullDevice];
    [task launchAndReturnError:nil];
}

- (void)togglePosting:(id)sender {
    self.postingToggleItem.enabled = NO;
    NSTask *task = [[NSTask alloc] init];
    task.executableURL = [NSURL fileURLWithPath:self.pythonPath];
    task.arguments = @[
        self.scriptPath,
        @"--state-dir", self.stateDirectory.path,
        @"set-posting", @"--enabled", self.postingEnabled ? @"0" : @"1",
    ];
    task.standardOutput = [NSFileHandle fileHandleWithNullDevice];
    task.standardError = [NSFileHandle fileHandleWithNullDevice];
    [task launchAndReturnError:nil];
    self.postingToggleItem.enabled = YES;
}

- (void)chooseRetryDelay:(id)sender {
    if (![self.currentStatus isEqualToString:@"error"]) return;
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Повтор публикации";
    alert.informativeText = @"Через сколько минут повторить?";
    [alert addButtonWithTitle:@"Сохранить"];
    [alert addButtonWithTitle:@"Отмена"];
    NSTextField *input = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 220, 24)];
    input.stringValue = @"1";
    alert.accessoryView = input;
    if ([alert runModal] != NSAlertFirstButtonReturn) return;
    NSInteger minutes = input.integerValue;
    if (minutes < 1 || minutes > 24 * 60) return;

    NSTask *task = [[NSTask alloc] init];
    task.executableURL = [NSURL fileURLWithPath:self.pythonPath];
    task.arguments = @[
        self.scriptPath,
        @"--state-dir", self.stateDirectory.path,
        @"set-retry", @"--minutes", [NSString stringWithFormat:@"%ld", (long)minutes],
    ];
    task.standardOutput = [NSFileHandle fileHandleWithNullDevice];
    task.standardError = [NSFileHandle fileHandleWithNullDevice];
    [task launchAndReturnError:nil];
}

- (void)openTable:(id)sender {
    [[NSWorkspace sharedWorkspace] openURL:
        [self.stateDirectory URLByAppendingPathComponent:@"combinations.csv"]];
}

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 4) return 2;
        NSURL *stateDirectory = [NSURL fileURLWithPath:@(argv[1]) isDirectory:YES];
        NSString *lockPath = [[stateDirectory URLByAppendingPathComponent:@"statusbar.lock"] path];
        int lockDescriptor = open(lockPath.fileSystemRepresentation, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
        if (lockDescriptor < 0 || flock(lockDescriptor, LOCK_EX | LOCK_NB) != 0) return 0;

        NSApplication *application = [NSApplication sharedApplication];
        StatusBarController *controller = [[StatusBarController alloc]
            initWithStateDirectory:stateDirectory
            pythonPath:@(argv[2])
            scriptPath:@(argv[3])];
        application.delegate = controller;
        [application run];
    }
    return 0;
}
