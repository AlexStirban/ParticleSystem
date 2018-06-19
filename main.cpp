#include <SFML\Graphics.hpp>

#include <vector>
#include <thread>
#include <array>
#include <mutex>
#include <atomic>
#include <random>
#include <cmath>


// Math ctes
static const double PI = 3.14159265359;


static const int MAX_BATCH_VERTS = 1000;
static const int MAX_PART_VEL = 30;
static const int PART_PER_CLICK = 100;

// Timestep
static const sf::Time dt = sf::seconds(1.f / 60.f);

// Window dimensions
static const sf::FloatRect W_DIM = sf::FloatRect(0, 0, 800, 600);


//Global variables (EVIL) used by worker threads
std::mutex m;
std::condition_variable cv;
std::atomic<int> threadSignals = 0;
int particleCounter = 0;




// Struct wich holds the velocities of every sf::Vertex correspoding to a particle
struct batch
{
	std::array<sf::Vertex,   MAX_BATCH_VERTS>    vertices;

	std::array<sf::Vector2f, MAX_BATCH_VERTS>    oldPositions;
	std::array<sf::Vector2f, MAX_BATCH_VERTS>    velocities;
	std::array<sf::Vector2f, MAX_BATCH_VERTS>    acceleration;
	int                                          population;

	batch() : population(0) {};
};


// Struct that the basic properties for a gravitational field
//		-origin: where the mass that would generate the field would be located at
//		-intensity: scalar value proportional to the mass that originates the field
//					and wich determines, partially, the strength of the attraction
struct gField
{
	sf::Vector2f origin;
	float        intensity;

	gField(sf::Vector2f _origin, float _inten) : origin(_origin), intensity(_inten) {};
};


// Holds the particles state, world dimensions, gravitational fields and the damping applied to simulate
// friction
struct world
{
	sf::FloatRect        bounds;
	std::vector<batch>   particles;
	std::vector<gField>  gravFields;
};



// ######################################################################################

// Spawns a particle with a given initial position and velocity
void spawnParticle (sf::Vector2f pos, sf::Vector2f vel, std::vector<batch>& batches)
{
	if (batches.empty())
		batches.emplace_back();

	int N = batches.back().population;


	if (N >= MAX_BATCH_VERTS)
	{
		batches.emplace_back();
		N = 0;
	}

	batches.back().vertices[N] = sf::Vertex(sf::Vertex(pos, sf::Color::Yellow));
	batches.back().oldPositions[N] = pos;
	batches.back().velocities[N] = vel;
	batches.back().population++;

	particleCounter++;
}





// Update particle movement
void update(world& env, int id)
{
	int MAX_THREADS = std::thread::hardware_concurrency();

	// Number of batches that each thread will handle
	int STEP = env.particles.size() / MAX_THREADS;
	int START = STEP * id;
	int END = (id == MAX_THREADS - 1) ? env.particles.size() : STEP * (id + 1);


	for (int j = START; j < END; j++)
	{
		for (int i = env.particles[j].population - 1; i >= 0; i--)
		{
			if (env.bounds.contains(env.particles[j].vertices[i].position))
			{
				// Verlet velocity
				env.particles[j].vertices[i].position +=  env.particles[j].velocities[i] * dt.asSeconds() + env.particles[j].acceleration[i] * dt.asSeconds() *  dt.asSeconds();

				// Gravitational field effects
				sf::Vector2f totalGForce;
				for (auto g : env.gravFields)
				{
					sf::Vector2f r = g.origin - env.particles[j].vertices[i].position;
					float r2 = r.x * r.x + r.y * r.y;

					totalGForce += r / r2 * g.intensity;		
				}

				env.particles[j].velocities[i] += 0.5f * (env.particles[j].acceleration[i] + totalGForce) * dt.asSeconds();
				env.particles[j].acceleration[i] = totalGForce;

			}
			else
			{
				env.particles[j].population--;
				particleCounter--;

				for (int m = i; m < env.particles[j].population; m++)
				{
					env.particles[j].vertices[m] = env.particles[j].vertices[m + 1];
					env.particles[j].velocities[m] = env.particles[j].velocities[m + 1];
				}	
			}
		}
	}

	std::lock_guard<std::mutex> lc(m);
	threadSignals++;
	cv.notify_all();
}

// Update the particle system
void updateSystem (world& env)
{
	int thNum = std::thread::hardware_concurrency();

	// Spwan threads, maybe would be better if we were usign worker threads???
	for (int i = 0; i < thNum; i++)
	{
		std::thread t(&update, std::ref(env), i);
		t.detach();
	}

	std::unique_lock<std::mutex> lock(m);
	cv.wait(lock, [thNum](){ return threadSignals == thNum; });
	threadSignals = 0;
}



int main()
{
	sf::RenderWindow window(sf::VideoMode(W_DIM.width, W_DIM.height), "PEngine");

	world env;

	std::default_random_engine gen;
	std::uniform_real_distribution<float> dis(0.0, 360.0);

	sf::Time time = sf::Time::Zero;
	sf::Clock clock;


	// Places a gravitational field with origin in the middle of the screen
	// Using negative values for the field intensity will produce a repulsive force
	env.gravFields.emplace_back(sf::Vector2f(400, 300), 500);

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		// Realtime input
		if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))
		{
			// get mouse current position
			sf::Vector2f pos = static_cast<sf::Vector2f> (sf::Mouse::getPosition(window));

			for (int i = 0; i < PART_PER_CLICK; i++)
			{
				float angle = dis(gen) * PI / 180;
				sf::Vector2f vel = sf::Vector2f(MAX_PART_VEL * cos(angle), -MAX_PART_VEL * sin(angle));

				spawnParticle(pos, vel, env.particles);
			}

		}


		// Using semi-fixed timestep to update de world
		time += clock.restart();
		while (time >= dt)
		{
			updateSystem(env);
			time -= dt;
		}


		window.clear(sf::Color::Black);

		// Draw batches of points
		for (auto& b : env.particles)
		{
			window.draw(&b.vertices[0], b.population, sf::Points);
		}

		window.display();
	}


	return 0;
}