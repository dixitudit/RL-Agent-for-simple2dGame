from random import randint
import cv2
import numpy as np
import pygame
pygame.init()

game_width = 600
game_height = 600
SPEED = -1

game_window = pygame.display.set_mode((game_width, game_height))
pygame.display.set_caption('Space Invader')
bg = [pygame.image.load('img\\bg1.jpg'), pygame.image.load('img\\bg2.jpg'),
      pygame.image.load('img\\bg3.jpg'), pygame.image.load('img\\bg4.jpg'),
      pygame.image.load('img\\bg5.jpg'), pygame.image.load('img\\bg6.jpg'),
      pygame.image.load('img\\bg7.jpg'), pygame.image.load('img\\bg8.jpg'),
      pygame.image.load('img\\bg9.jpg'), pygame.image.load('img\\bg10.jpg'),
      pygame.image.load('img\\bg11.jpg'), pygame.image.load('img\\bg12.jpg')]

ship1 = pygame.image.load('img\\player.png')
ship2 = pygame.image.load('img\\enemy.png')
life = pygame.image.load('img\\life.png')
fire = pygame.mixer.Sound('sounds\\bulletFire.wav')
blast = pygame.mixer.Sound('sounds\\blast.wav')
lostLife = pygame.mixer.Sound('sounds\\lifeLost.wav')
music1 = pygame.mixer.music.load('sounds\\music.mp3')
# pygame.mixer.music.play(-1)
clock = pygame.time.Clock()


class Player:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vel = 6 + SPEED
        self.hitBox = (self.x + 5, self.y + 5, self.width - 10, self.height - 10)
        self.lifeCount = 3
        self.score = 0

    def draw(self, surface):
        for i in range(self.lifeCount):
            game_window.blit(life, (30 + (i * 40), 30))
        game_window.blit(ship1, (self.x, self.y))
        surface.blit(ship1, (self.x, self.y))
        self.hitBox = (self.x + 5, self.y + 5, self.width - 10, self.height - 10)


class Enemy:
    def __init__(self, max_x, y, width, height):
        self.x = randint(0, max_x)
        self.y = y
        self.width = width
        self.height = height
        self.vel = 4 + SPEED
        self.hitBox = (self.x + 5, self.y + 5, self.width - 10, self.height - 10)
        self.isVisible = True
        self.deathByBullet = False

    def draw_enemy(self, surface):
        if self.isVisible:
            game_window.blit(ship2, (self.x, self.y))
            surface.blit(ship2, (self.x, self.y))
            self.y += self.vel
            self.hitBox = (self.x + 5, self.y + 5, self.width - 10, self.height - 10)


class Projectile:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.vel = 8 + SPEED

    def draw_bullet(self, surface):
        pygame.draw.circle(game_window, self.color, (self.x, self.y), self.radius)
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)

class SpaceInvaderAI:
    def __init__(self):
        self.guardian = Player(game_width // 2 - 32, game_height - 90, 64, 71)
        self.bullets = []
        self.bg_count = 0
        self.shoot_loop = 0
        self.intro = 0
        self.invader = Enemy(game_width - 64, 0, 64, 64)
        self.font = pygame.font.SysFont('comicsans', 20, True)
        self.frame_iteration = 0
        self.state_surface = pygame.Surface((game_width, game_height))
        self.draw_game_window()
        self.clock = pygame.time.Clock()

 
    def draw_game_window(self):
        # Create a surface to render the game state
        # surface is for creating an image for DQN training and game window is actual game window
        self.state_surface = pygame.Surface((game_width, game_height))  # Create a surface with the dimensions of the game window
        
        if self.bg_count >= 36:
            self.bg_count = 0
        game_window.blit(bg[self.bg_count // 3], (0, 0)) 
        self.state_surface.fill((0, 0, 0))  # Fill the surface with black background
        self.bg_count += 1
        self.guardian.draw(self.state_surface)
        file2 = open('textforhc\\highScore.txt', 'r')
        high_score = int(file2.read())
        file2.close()
        text_h_score = self.font.render('High Score: ' + str(high_score), 1, (120, 0, 120))
        game_window.blit(text_h_score, (game_width - text_h_score.get_width() - 10, 10))
        if high_score < self.guardian.score:
            score_color = (0, 120, 0)
        else:
            score_color = (120, 120, 120)
        text = self.font.render('Score: ' + str(self.guardian.score), 1, score_color)
        game_window.blit(text, (game_width - text.get_width()-10, 30))
        for rounds in self.bullets:
            rounds.draw_bullet(self.state_surface)
        self.invader.draw_enemy(self.state_surface)
        pygame.draw.line(self.state_surface, (200, 0, 0), (0, game_height), (game_width, game_height), 8)
        pygame.draw.line(game_window, (200, 0, 0), (0, game_height), (game_width, game_height), 8)
        pygame.display.update()
        # Convert the surface to an image
        state_image = pygame.image.tostring(self.state_surface, 'RGB')
        
        # Convert image to numpy array for further processing
        state_array = np.frombuffer(state_image, dtype=np.uint8)
        state_array = state_array.reshape((game_height, game_width, 3))

        # # Display the game state image
        # cv2.imshow('Game State', state_array)  # Show the game state image
        # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        # cv2.destroyAllWindows()  # Close the window when a key is pressed

        if self.guardian.lifeCount == 0:
            self.bullets = []
            file = open('textforhc\\highScore.txt', 'r')
            if self.guardian.score > int(file.read()):
                file1 = open('textforhc\\highScore.txt', 'w')
                file1.write(str(self.guardian.score))
                file1.close()
            file.close()
            text2 = self.font.render('Game Over! Restarting...', 1, (120, 0, 0))
            game_window.blit(text2, (game_width//2 - text2.get_width()//2, game_height//2))
            pygame.display.update()
            pygame.time.delay(2)
            self.guardian.__init__(game_width // 2 - 32, game_height - 90, 64, 71)
            return state_array
        if self.guardian.lifeCount == 3 and self.intro == 0:
            text2 = self.font.render("""SPACE INVADER. Starting...""", 1, (100, 0, 120))
            game_window.blit(bg[0], (0, 0))
            game_window.blit(text2, (game_width//2 - text2.get_width()//2, game_height//2))
            pygame.display.update()
            pygame.time.delay(2)
            self.intro = 1
        return state_array

    def handle_bullets(self):
        for bullet in self.bullets:
            if self.invader.hitBox[1] + self.invader.hitBox[3] - 10 > bullet.y > self.invader.hitBox[1] and \
                    self.invader.isVisible:
                if self.invader.hitBox[0] < bullet.x < self.invader.hitBox[2] + self.invader.hitBox[0]:
                    self.invader.isVisible = False
                    self.invader.deathByBullet = True
                    self.guardian.score += 5
                    blast.play()
                    self.bullets.pop(self.bullets.index(bullet))
            if bullet.y >= 0:
                bullet.y -= bullet.vel
            else:
                self.bullets.pop(self.bullets.index(bullet))


    def handle_collision(self):
        # if self.guardian.hitBox[1] < self.invader.hitBox[1] + self.invader.hitBox[3] \
        #     < self.guardian.hitBox[1] + self.guardian.hitBox[3]:
        #     if self.guardian.hitBox[0] < self.invader.hitBox[0] < self.guardian.hitBox[0] + self.guardian.hitBox[2] \
        #         or self.guardian.hitBox[0] < self.invader.hitBox[0] + self.invader.hitBox[2] <\
        #             self.guardian.hitBox[0] + self.guardian.hitBox[2]:
        #         self.invader.isVisible = False
        #         # lostLife.play()
        #         self.guardian.x = game_width // 2 - 32
        #         self.guardian.y = game_height - 90
        #         self.guardian.lifeCount -= 1
        #         return -1
        if self.invader.hitBox[1] > game_height:
            self.invader.isVisible = False
            lostLife.play()
            self.guardian.x = game_width // 2 - 32
            self.guardian.y = game_height - 90
            self.guardian.lifeCount -= 1
            return -1
        return 0


    def handle_keypress(self, action):
        # action
        # [1, 0, 0, 0] -> Left
        # [0, 1, 0, 0] -> Fire
        # [0, 0, 1, 0] -> Right
        # [0, 0, 0, 1] -> Nothing

        if action == None:
            keys = pygame.key.get_pressed()

            if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and self.guardian.x + self.guardian.width + self.guardian.vel < game_width:
                self.guardian.x += self.guardian.vel
            elif (keys[pygame.K_LEFT] or keys[pygame.K_a])and self.guardian.x - self.guardian.vel > 0:
                self.guardian.x -= self.guardian.vel

            if (keys[pygame.K_UP] or keys[pygame.K_w])and self.shoot_loop == 0:
                if len(self.bullets) < 6:
                    fire.play()
                    self.bullets.append(Projectile(self.guardian.x + self.guardian.width // 2, self.guardian.y, 6, (120, 0, 0)))

                self.shoot_loop = 1
        else:
            if action == 0 and self.guardian.x - self.guardian.vel > 0:
                self.guardian.x -= self.guardian.vel
            elif action == 2 and self.guardian.x + self.guardian.width + self.guardian.vel < game_width:
                self.guardian.x += self.guardian.vel
            elif action == 1 and self.shoot_loop == 0:
                if len(self.bullets) < 6:
                    fire.play()
                    self.bullets.append(Projectile(self.guardian.x + self.guardian.width // 2, self.guardian.y, 6, (120, 0, 0)))
                self.shoot_loop = 1

    def play_step(self, action=None):
        self.frame_iteration += 1
        done = False
        reward = 0
        if self.shoot_loop > 0:
            self.shoot_loop += 1
        if self.shoot_loop > 4:
            self.shoot_loop = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                done = True
                return reward, done, self.guardian.score
            
        if not self.invader.isVisible:
            if self.invader.deathByBullet:
                reward = 1
            self.invader.draw_enemy(self.state_surface)
            self.invader.__init__(game_width - 64, 0, 64, 64)
        self.handle_bullets()
        x=self.handle_collision()
        if not x==0:
            reward = x
        self.handle_keypress(action)
        if(self.guardian.lifeCount==0):
            done = True
        return reward, done, self.guardian.score




if __name__ == '__main__':
    game = SpaceInvaderAI()
    while True:
        game.clock.tick(36)
        act = randint(0, 3)
        reward, done, score = game.play_step()
        stateNew = game.draw_game_window()
        # print(stateNew.shape)
        # if reward !=0:
        #     print(reward)
        # if done:
        #     break