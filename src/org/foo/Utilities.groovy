package org.foo
class Utilities implements Serializable {
	def steps

	Utilities(steps) {
		this.steps = steps
	}
	def mvn(args) {
		return this.steps.env.HOME
	}
	
	static def mvn(step) {
		return step.env.BRANCH_NAME
	}

}
