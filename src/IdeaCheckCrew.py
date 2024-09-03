from textwrap import dedent
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from tools import ExaSearchToolset
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

#Call Gemini model
llm1 = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro',
    verbose=True,
    temperature=0.8,
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

# llm2 = ChatGroq(
#     api_key=os.getenv('GROQ_API_KEY'),
#     model="mixtral-8x7b-32768"
# )


class IdeaCheckAgents:
    def idea_research_agent(self):
        return Agent(
            role="Expert Business Analyst",
            goal="Find out if any other business or individual has already implemented the given idea",
            tools=ExaSearchToolset.tools(),
            backstory=dedent("""\
                You are an expert business analyst with a passion for researching new ideas.
                Your expertise lies in finding out if any other business or individual has already
                implemented a given idea. You are skilled in market research, competitor analysis,
                and identifying market trends."""),
            verbose=True,
            llm=llm1,
            #max_iter=3
        )
    
    def deciding_agent(self):
        return Agent(
            role="Next Step Decider",
            goal="Decide if there is another company that has the same idea as the user",
            backstory=dedent("""\
                You are a decision maker with a keen eye for detail. Your expertise lies in
                analyzing information and making informed decisions. You are skilled in
                identifying patterns, trends, and opportunities."""),
            verbose=True,
            llm=llm1,
            #max_iter= 3
        )
    
    def next_steps_agent(self):
        return Agent(
            role="Market Research Expert",
            goal="Provide the next steps that the user should take to implement the idea",
            tools=ExaSearchToolset.tools(),
            backstory=dedent("""\
                You are a market research expert with a knack for identifying opportunities.
                Your expertise lies in analyzing market trends, identifying potential customers,
                and developing strategies to reach them. You are skilled in market research,
                competitor analysis, and strategic planning. You are passionate about helping
                businesses succeed and grow."""),
            verbose=True,
            llm=llm1,
            #max_iter= 3
        )

class IdeaCheckTasks:
    def idea_research_task(self, agent, idea):
        return Task(
            description=dedent(f"""\
                Your task is to find out if any other business or individual 
                has already implemented the given idea. Be slightly more general 
                with the idea to get better results as the EXACT idea may not have
                been done but something similar may constitute the same idea. (Example: 
                "AI Advice Service" pretty much the same as AI consulting and strategy 
                development)

                Idea: {idea}"""),
            expected_output=dedent("""\
                If the exact idea has been done before, provide and name/s 
                of the companies or individuals that have implemented it. (If there 
                are more than 5 companies, provide the top 5 companies using the idea)
                If the Idea has not be done before then provide a summary of the 
                market fit and viability of the idea."""),
            agent=agent,
        )
    
    def deciding_task(self, agent):
        return Task(
            description=dedent(f"""\
                Based on the output of the previous agent, you will decide on whether 
                the idea has been done or not. You can generally tell if the idea has 
                been done before if the agent provides the names of companies that are
                already using the idea."""),
            expected_output=dedent("""If the idea has been done before, then you will
                respond with 'done'. If the idea has not been done before, then you will 
                respond with 'not done'"""),
            agent=agent, 
        )

    def next_steps_task(self, agent, idea):
        return Task(
            description=dedent(f"""\
                Using the idea provided by the user, come up with the next steps that they
                should take to implement the idea. This could be a list of tasks or a full
                plan of action.
                               
                Idea: {idea}"""),
            expected_output=dedent("""\
                A set of next steps that the user should take to implement the idea. This 
                should include: who to talk to, how to make the product and how to get funding"""),
            agent=agent,
        )
    
    
    
def run():
    idea = input("Enter in your idea\n")

    tasks = IdeaCheckTasks()
    agents = IdeaCheckAgents()

    # Create agent
    idea_research_agent = agents.idea_research_agent()
    deciding_agent = agents.deciding_agent()
    next_steps_agent = agents.next_steps_agent()

    # Create tasks
    idea_research_task = tasks.idea_research_task(idea_research_agent, idea)

    # Create Crew
    crew1 = Crew(
        agents=[idea_research_agent],
        tasks=[idea_research_task]
    )

    result1 = crew1.kickoff()
    print("result from crew1")
    print(result1)


    deciding_task = tasks.deciding_task(agents.deciding_agent())
    deciding_task.context = [idea_research_task]
    crew2 = Crew(
        agents=[deciding_agent],
        tasks=[deciding_task]
    )
    
    if (crew2.kickoff() == "done"):
        idea += input("""Since another organization has already implemented the idea, please enter a change/improvement to the idea:\n""")

        
    next_steps_task = tasks.next_steps_task(next_steps_agent, idea)
    next_steps_task.context = [idea_research_task]

    crew3 = Crew(
        agents=[next_steps_agent],
        tasks=[next_steps_task]
    )

    print(crew3.kickoff())

if __name__ == '__main__':
    run()
